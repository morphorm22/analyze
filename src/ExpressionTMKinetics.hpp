#ifndef PLATO_EXPRESSION_TMKINETICS_HPP
#define PLATO_EXPRESSION_TMKINETICS_HPP

#include "AbstractTMKinetics.hpp"
#include "ExpressionEvaluator.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/******************************************************************************/
/*! Expression Thermomechanics Kinetics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class ExpressionTMKinetics :
    public Plato::AbstractTMKinetics<EvaluationType, SimplexPhysics>
{
protected:
    static constexpr int TDofOffset = EvaluationType::SpatialDim;
    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using KineticsScalarType = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using KinematicsScalarType = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>; /*!<   strain variables automatic differentiation type */
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using Plato::SimplexThermomechanics<EvaluationType::SpatialDim>::mNumVoigtTerms;
    using PhysicsType = typename Plato::SimplexThermomechanics<EvaluationType::SpatialDim>;
    using PhysicsType::mNumDofsPerNode;

    Plato::TensorConstant<EvaluationType::SpatialDim> mThermalExpansivityConstant;
    Plato::TensorConstant<EvaluationType::SpatialDim> mThermalConductivityConstant;
    Plato::Scalar mRefTemperature;
    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;
    Plato::VoigtMap<EvaluationType::SpatialDim> cVoigtMap;

    std::string mExpression;
    Plato::Scalar mE0;
    KineticsScalarType mPoissonsRatio;
    ControlScalarType mControlValue;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model
    **********************************************************************************/
    ExpressionTMKinetics(const Teuchos::RCP<Plato::MaterialModel<EvaluationType::SpatialDim>> aMaterialModel) :
            AbstractTMKinetics<EvaluationType, SimplexPhysics>(aMaterialModel),
            mRefTemperature(aMaterialModel->getScalarConstant("Reference Temperature")),
            mScaling(aMaterialModel->getScalarConstant("Temperature Scaling")),
            mScaling2(mScaling*mScaling),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
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
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0,aControl.extent(0)), LAMBDA_EXPRESSION(Plato::OrdinalType i)
            {
                for(Plato::OrdinalType j=0; j<aControl.extent(1); j++)
                {
                    aLocalControl(i,j) = tControlValue;
                }
            },"Compute local control");
        }
        else
        {
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0,aControl.extent(0)), LAMBDA_EXPRESSION(Plato::OrdinalType i)
            {
                for(Plato::OrdinalType j=0; j<aControl.extent(1); j++)
                {
                    aLocalControl(i,j) = aControl(i,j);
                }
            },"Compute local control");
        }
    }

    void
    calculateElementYoungsModulusValues(const Plato::OrdinalType &aNumCells,
                                        const Plato::ScalarMultiVectorT<ControlScalarType> &aLocalControl,
                                        Plato::ScalarMultiVectorT<KineticsScalarType> &aElementYoungsModulusValues) const
    {
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Plato::InterpolateFromNodal<EvaluationType::SpatialDim, 1, 0> tInterpolateFromNodal;
        Plato::ScalarVectorT<ControlScalarType> tElementDensity("Gauss point density", aNumCells);

        ExpressionEvaluator<Plato::ScalarMultiVectorT<KineticsScalarType>,
                            Plato::ScalarMultiVectorT<KinematicsScalarType>,
                            Plato::ScalarVectorT<ControlScalarType>,
                            Plato::Scalar > tExpEval;
        
        tExpEval.parse_expression(mExpression.c_str());
        tExpEval.setup_storage(aNumCells, 1);
        tExpEval.set_variable("E0", mE0);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,aNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            // Calculate the node-averaged density for the element/cell
            tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aLocalControl, tElementDensity);

            tExpEval.set_variable("tElementDensity", tElementDensity, aCellOrdinal);
            tExpEval.evaluate_expression( aCellOrdinal, aElementYoungsModulusValues );
        },"Compute Youngs Modulus for each Element");
        Kokkos::fence();
        tExpEval.clear_storage();
    }

    void
    computeThermalStrainStressAndFlux(const Plato::OrdinalType &aNumCells,
                                      Kokkos::View<StateT*, Plato::MemSpace> const& aTemperature,
                                      const Plato::ScalarMultiVectorT<KineticsScalarType> &aElementYoungsModulusValues,
                                      Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aStrain,
                                      Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aStress,
                                      Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aFlux,
                                      Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aTGrad) const
    {
        auto tScaling = mScaling;
        auto tScaling2 = mScaling2;
        auto tRefTemperature = mRefTemperature;
        auto& tThermalExpansivityConstant = mThermalExpansivityConstant;
        auto& tThermalConductivityConstant = mThermalConductivityConstant;
        auto& tVoigtMap = cVoigtMap;
        auto tPoissonsRatio = mPoissonsRatio;

        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, aNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
        {
            StateT tTemperature = aTemperature(aCellOrdinal);
            auto tCurYoungsModulus = aElementYoungsModulusValues(aCellOrdinal,0);
            Plato::IsotropicStiffnessConstant<EvaluationType::SpatialDim, KineticsScalarType> 
                    tStiffnessConstant(tCurYoungsModulus, tPoissonsRatio);            
            
            // compute thermal strain
            //
            StateT tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<EvaluationType::SpatialDim; iDim++ ){
                tstrain[iDim] = tScaling * tThermalExpansivityConstant(tVoigtMap.I[iDim], tVoigtMap.J[iDim])
                            * (tTemperature - tRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(aCellOrdinal,iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(aCellOrdinal,iVoigt) += (aStrain(aCellOrdinal,jVoigt)-tstrain[jVoigt])*tStiffnessConstant(iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<EvaluationType::SpatialDim; iDim++){
                aFlux(aCellOrdinal,iDim) = 0.0;
                for( int jDim=0; jDim<EvaluationType::SpatialDim; jDim++){
                    aFlux(aCellOrdinal,iDim) += tScaling2 * aTGrad(aCellOrdinal,jDim)*tThermalConductivityConstant(iDim, jDim);
                }
            }
        }, "Cauchy stress");
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
    operator()( Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aStress,
                Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aFlux,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aStrain,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aTGrad,
                Kokkos::View<StateT*,       Plato::MemSpace> const& aTemperature,
                const Plato::ScalarMultiVectorT <ControlScalarType> & aControl) const override
    {
        const Plato::OrdinalType tNumCells = aStrain.extent(0);

        Plato::ScalarMultiVectorT<ControlScalarType> tLocalControl("Local Control", aControl.extent(0), aControl.extent(1));

        // Set local control to user-defined value if requested.
        setLocalControl(aControl, tLocalControl);

        // Calculate a Youngs Modulus for each element based on its density.
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementYoungsModulusValues("Element Youngs Modulus", tNumCells, 1);
        calculateElementYoungsModulusValues(tNumCells, tLocalControl, tElementYoungsModulusValues);

        computeThermalStrainStressAndFlux(tNumCells, aTemperature, tElementYoungsModulusValues, aStrain, aStress, aFlux, aTGrad);
    }
};
// class ExpressionTMKinetics



}// namespace Plato
#endif

/*
#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 3)
#endif
*/
