#ifndef PLATO_EXPRESSION_TMKINETICS_HPP
#define PLATO_EXPRESSION_TMKINETICS_HPP

#include "AbstractTMKinetics.hpp"
#include "ExpressionEvaluator.hpp"
#include "InterpolateFromNodal.hpp"

namespace Plato
{

/******************************************************************************/
/*! Expression Thermomechanics Kinetics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<typename EvaluationType, typename ElementType>
class ExpressionTMKinetics :
    public Plato::AbstractTMKinetics<EvaluationType, ElementType>
{
protected:
    using StateT  = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
    using KinematicsScalarType = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumSpatialDims;

    Plato::TensorConstant<mNumSpatialDims> mThermalExpansivityConstant;
    Plato::TensorConstant<mNumSpatialDims> mThermalConductivityConstant;

    Plato::Scalar mRefTemperature;
    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;
    Plato::VoigtMap<mNumSpatialDims> cVoigtMap;

    std::string mExpression;
    Plato::Scalar mE0;
    KineticsScalarType mPoissonsRatio;
    ControlScalarType mControlValue;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model
    **********************************************************************************/
    ExpressionTMKinetics(const Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> aMaterialModel) :
            AbstractTMKinetics<EvaluationType, ElementType>(aMaterialModel),
            mRefTemperature(aMaterialModel->getScalarConstant("Reference Temperature")),
            mScaling(aMaterialModel->getScalarConstant("Temperature Scaling")),
            mScaling2(mScaling*mScaling)
    {
        mThermalExpansivityConstant = aMaterialModel->getTensorConstant("Thermal Expansivity");
        mThermalConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");
        mE0 = aMaterialModel->getScalarConstant("E0");
        mExpression = aMaterialModel->expression();
        mPoissonsRatio = aMaterialModel->getScalarConstant("Poissons Ratio");
        mControlValue = -1.0;
        if(aMaterialModel->scalarConstantExists("Density"))
        {
            mControlValue = aMaterialModel->getScalarConstant("Density");
        }
    }

    void 
    setLocalControl(const Plato::ScalarMultiVectorT <ControlScalarType> &aControl,
                               Plato::ScalarMultiVectorT<ControlScalarType> &aLocalControl) const
    {
        // This code allows for the user to specify a global density value for all nodes when 
        // running a forward problem (when mControlValue != -1.0). This is set with a "Density" entry in 
        // the input deck (see parsing of this in MaterialModel constructor). Typically, though, the passed in 
        // control will just be used. 
        if(mControlValue != -1.0)
        {
            auto tControlValue = mControlValue;
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0,aControl.extent(0)), KOKKOS_LAMBDA(Plato::OrdinalType i)
            {
                for(Plato::OrdinalType j=0; j<aControl.extent(1); j++)
                {
                    aLocalControl(i,j) = tControlValue;
                }
            },"Compute local control");
        }
        else
        {
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0,aControl.extent(0)), KOKKOS_LAMBDA(Plato::OrdinalType i)
            {
                for(Plato::OrdinalType j=0; j<aControl.extent(1); j++)
                {
                    aLocalControl(i,j) = aControl(i,j);
                }
            },"Compute local control");
        }
    }

    void
    calculateYoungsModulusValues(
        Plato::OrdinalType                            const & aNumCells,
        Plato::ScalarMultiVectorT<ControlScalarType>  const & aLocalControl,
        Plato::ScalarMultiVectorT<KineticsScalarType>       & aElementYoungsModulusValues
    ) const
    {
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Plato::InterpolateFromNodal<ElementType, 1, 0> tInterpolateFromNodal;
        Plato::ScalarVectorT<ControlScalarType> tElementDensity("Gauss point density", aNumCells*tNumPoints);

        ExpressionEvaluator<Plato::ScalarMultiVectorT<KineticsScalarType>,
                            Plato::ScalarMultiVectorT<KinematicsScalarType>,
                            Plato::ScalarVectorT<ControlScalarType>,
                            Plato::Scalar > tExpEval;
        
        tExpEval.parse_expression(mExpression.c_str());
        tExpEval.setup_storage(aNumCells*tNumPoints, 1);
        tExpEval.set_variable("E0", mE0);

        Kokkos::parallel_for("compute element density", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {aNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            // Calculate the node-averaged density for the element/cell
            auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            tElementDensity(tEntryOrdinal) = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aLocalControl);
        });

        tExpEval.set_variable("tElementDensity", tElementDensity);
        Kokkos::parallel_for("compute youngs modulus", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {aNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;

            tExpEval.evaluate_expression( tEntryOrdinal, aElementYoungsModulusValues );
        });
        Kokkos::fence();
        tExpEval.clear_storage();
    }

    void
    computeThermalStrainStressAndFlux(
        Plato::OrdinalType                            const & aNumCells,
        Plato::ScalarMultiVectorT<StateT>             const & aTemperature,
        Plato::ScalarMultiVectorT<KineticsScalarType> const & aElementYoungsModulusValues,
        Plato::ScalarArray3DT<KinematicsScalarType>   const & aStrain,
        Plato::ScalarArray3DT<KineticsScalarType>     const & aStress,
        Plato::ScalarArray3DT<KineticsScalarType>     const & aFlux,
        Plato::ScalarArray3DT<KinematicsScalarType>   const & aTGrad
    ) const
    {
        auto tScaling = mScaling;
        auto tScaling2 = mScaling2;
        auto tRefTemperature = mRefTemperature;
        auto& tThermalExpansivityConstant = mThermalExpansivityConstant;
        auto& tThermalConductivityConstant = mThermalConductivityConstant;
        auto& tVoigtMap = cVoigtMap;
        auto tPoissonsRatio = mPoissonsRatio;

        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute element kinematics", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {aNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            StateT tTemperature = aTemperature(iCellOrdinal, iGpOrdinal);

            auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            auto tCurYoungsModulus = aElementYoungsModulusValues(tEntryOrdinal, 0);
            Plato::IsotropicStiffnessConstant<mNumSpatialDims, KineticsScalarType> tStiffnessConstant(tCurYoungsModulus, tPoissonsRatio);            
            // compute thermal strain
            //
            StateT tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<mNumSpatialDims; iDim++ ){
                tstrain[iDim] = tScaling * tThermalExpansivityConstant(tVoigtMap.I[iDim], tVoigtMap.J[iDim])
                            * (tTemperature - tRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(iCellOrdinal, iGpOrdinal, iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(iCellOrdinal, iGpOrdinal, iVoigt) += (aStrain(iCellOrdinal, iGpOrdinal, jVoigt)-tstrain[jVoigt])*tStiffnessConstant(iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<EvaluationType::SpatialDim; iDim++){
                aFlux(iCellOrdinal, iGpOrdinal, iDim) = 0.0;
                for( int jDim=0; jDim<mNumSpatialDims; jDim++){
                    aFlux(iCellOrdinal, iGpOrdinal, iDim) += tScaling2 * aTGrad(iCellOrdinal, iGpOrdinal, jDim)*tThermalConductivityConstant(iDim, jDim);
                }
            }
        });
    }

    /***********************************************************************************
     * \brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    void
    operator()(
        Plato::ScalarArray3DT<KineticsScalarType>    const & aStress,
        Plato::ScalarArray3DT<KineticsScalarType>    const & aFlux,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aStrain,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aTGrad,
        Plato::ScalarMultiVectorT<StateT>            const & aTemperature,
        Plato::ScalarMultiVectorT<ControlScalarType> const & aControl
    ) const override
    {
        const Plato::OrdinalType tNumCells = aStrain.extent(0);

        Plato::ScalarMultiVectorT<ControlScalarType> tLocalControl("Local Control", aControl.extent(0), aControl.extent(1));

        // Set local control to user-defined value if requested.
        setLocalControl(aControl, tLocalControl);

        auto tNumPoints = ElementType::getCubWeights().size();

        // Calculate a Youngs Modulus for each element based on its density.
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementYoungsModulusValues("Youngs Modulus", tNumCells*tNumPoints, 1);
        calculateYoungsModulusValues(tNumCells, tLocalControl, tElementYoungsModulusValues);

        computeThermalStrainStressAndFlux(tNumCells, aTemperature, tElementYoungsModulusValues, aStrain, aStress, aFlux, aTGrad);
    }
};// class ExpressionTMKinetics
}// namespace Plato
#endif
