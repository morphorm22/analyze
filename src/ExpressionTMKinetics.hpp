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

    Plato::Rank4VoigtConstant<EvaluationType::SpatialDim> mElasticStiffnessConstant;
    Plato::TensorConstant<EvaluationType::SpatialDim> mThermalExpansivityConstant;
    Plato::TensorConstant<EvaluationType::SpatialDim> mThermalConductivityConstant;
    Plato::Scalar mRefTemperature;
    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;
    Plato::VoigtMap<EvaluationType::SpatialDim> cVoigtMap;

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
        mElasticStiffnessConstant = aMaterialModel->getRank4VoigtConstant("Elastic Stiffness");
        mThermalExpansivityConstant = aMaterialModel->getTensorConstant("Thermal Expansivity");
        mThermalConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");
    }

    /***********************************************************************************
     * \brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    //template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    void
    operator()( Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aStress,
                Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aFlux,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aStrain,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aTGrad,
                Kokkos::View<StateT*,       Plato::MemSpace> const& aTemperature,
                const Plato::ScalarMultiVectorT <ControlScalarType> & aControl) const override
    {
        auto tScaling = mScaling;
        auto tScaling2 = mScaling2;
        auto tRefTemperature = mRefTemperature;
        auto& tThermalExpansivityConstant = mThermalExpansivityConstant;
        auto& tThermalConductivityConstant = mThermalConductivityConstant;
        auto& tElasticStiffnessConstant = mElasticStiffnessConstant;
        auto& tVoigtMap = cVoigtMap;
        const Plato::OrdinalType tNumCells = aStrain.extent(0);

        Plato::InterpolateFromNodal<EvaluationType::SpatialDim, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;
        Plato::ScalarVectorT<ControlScalarType> tElementDensity("Gauss point density", tNumCells);
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();

        // Calculate the node-averaged density for the element/cell
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aControl, tElementDensity);
        },"Compute Element Densities");

        // Calculate Youngs Modulus for each element based on the element density
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementYoungsModulusValues("Element Youngs Modulus", tNumCells, 1);

        ExpressionEvaluator< Plato::ScalarMultiVectorT<KineticsScalarType>,
                            Plato::ScalarMultiVectorT<KinematicsScalarType>,
                            Plato::ScalarVectorT<Plato::Scalar>,
                            Plato::Scalar > tExpEval;
        
        tExpEval.parse_expression("E0*tElementDensity*tElementDensity*tElementDensity");
        tExpEval.setup_storage(tNumCells, 1);
        tExpEval.set_variable("E0", 1e9);
    
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            tExpEval.set_variable("tElementDensity", tElementDensity, aCellOrdinal);
            tExpEval.evaluate_expression( aCellOrdinal, tElementYoungsModulusValues );
        },"Compute Youngs Modulus for each Element");






/*
        ExpressionEvaluator< Plato::ScalarMultiVectorT<KineticsScalarType>,
                            Plato::ScalarMultiVectorT<KinematicsScalarType>,
                            Plato::ScalarVectorT<Plato::Scalar>,
                            Plato::Scalar > tExpEval;

        tExpEval.set_variable("element_averaged_rho", tElementAveragedRho);

        // Use an expression evaluator to calculate Youngs Modulus for each element
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells),
                            LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute the stress.  This loop cannot be parallelized
            // because the cell stiffness is set locally and is used by
            // all threads. In other words the tCellStiffness[tVoigtIndex_I]
            // is in shared memory and used by all threads
            for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
            {
                // Values that change based on the tVoigtIndex_I index.
                if( tVarMaps(cCellStiffness).key )
                tExpEval.set_variable( "rho", tRho
                                        tCellStiffness[tVoigtIndex_I],
                                        aCellOrdinal );

                // Evaluate the expression for this cell. Note: the second
                // index of tStress is over tVoigtIndex_J.
                tExpEval.evaluate_expression( aCellOrdinal, tStress );

                // Sum the stress values.
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;

                for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
                {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) += tStress(aCellOrdinal, tVoigtIndex_J);

                // The original stress equation.
                // aCauchyStress(aCellOrdinal, tVoigtIndex_I) += (aSmallStrain(aCellOrdinal, tVoigtIndex_J)
                // - tReferenceStrain(tVoigtIndex_J)) * tCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
                }
            }
        }, "Youngs Modulus Calculation" );


*/

        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
        {
            StateT tTemperature = aTemperature(aCellOrdinal);
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
                    aStress(aCellOrdinal,iVoigt) += (aStrain(aCellOrdinal,jVoigt)-tstrain[jVoigt])*tElasticStiffnessConstant(iVoigt, jVoigt);
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
