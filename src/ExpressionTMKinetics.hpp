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

//    Plato::Rank4VoigtConstant<EvaluationType::SpatialDim> mElasticStiffnessConstant;
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
 //       mElasticStiffnessConstant = aMaterialModel->getRank4VoigtConstant("Elastic Stiffness");
        mThermalExpansivityConstant = aMaterialModel->getTensorConstant("Thermal Expansivity");
        mThermalConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");
        mE0 = aMaterialModel->getScalarConstant("E0");
        mExpression = aMaterialModel->expression();
        mPoissonsRatio = aMaterialModel->getScalarConstant("Poissons Ratio");
        mControlValue = aMaterialModel->getScalarConstant("Density");
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
//        auto& tElasticStiffnessConstant = mElasticStiffnessConstant;
        auto& tVoigtMap = cVoigtMap;
        const Plato::OrdinalType tNumCells = aStrain.extent(0);

        Plato::InterpolateFromNodal<EvaluationType::SpatialDim, 1, 0> tInterpolateFromNodal;
        Plato::ScalarVectorT<ControlScalarType> tElementDensity("Gauss point density", tNumCells);
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();

//Plato::ScalarMultiVectorT <ControlScalarType> tControlCopy = aControl;
//std::cout << "Extent0: " << tControlCopy.extent(0) << std::endl;
//std::cout << "Extent1: " << tControlCopy.extent(1) << std::endl;
//for(size_t i=0; i<tControlCopy.extent(0); ++i)
//{
//    tControlCopy(i,0) = .75;
//    std::cout << "Control: " << tControlCopy(i,0) << std::endl;
 //   std::cout << tElementYoungsModulusValues(i,0) << std::endl;
//}

printf("Local Control Values\n");
        auto tControlValue = mControlValue;
        Plato::ScalarMultiVectorT<ControlScalarType> tLocalControl("Local Control", aControl.extent(0), aControl.extent(1));
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,aControl.extent(0)), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            for(int j=0; j<aControl.extent(1); ++j)
            {
                tLocalControl(aCellOrdinal, j) = tControlValue;
                printf("%lf ", tLocalControl(aCellOrdinal, j));
            }
            printf("\n");
        },"Set Local Control");


/*
printf("Control Values\n");
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            for(int j=0; j<4; ++j)
                printf("%lf ", aControl(aCellOrdinal, j));
            printf("\n");
        },"Compute Element Densities");
*/


printf("Element Density Values\n");
        // Calculate the node-averaged density for the element/cell
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, tLocalControl, tElementDensity);
            printf("%lf\n", tElementDensity(aCellOrdinal));
        },"Compute Element Densities");

/*
FILE *fp2=fopen("Control.txt", "w");
FILE *fp=fopen("ElemDensities.txt", "w");
std::cout << "Element Densities" << std::endl;
for(int i=0; i<tNumCells; ++i)
{
    fprintf(fp, "%lf\n", tElementDensity(i));
    std::cout << tElementDensity(i) << std::endl;
    for(int j=0; j<4; ++j)
        fprintf(fp2, "%lf ", aControl(i, j));
    fprintf(fp2, "\n");
}
fclose(fp);
fclose(fp2);
*/
        // Calculate Youngs Modulus for each element based on the element density
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementYoungsModulusValues("Element Youngs Modulus", tNumCells, 1);

        ExpressionEvaluator<Plato::ScalarMultiVectorT<KineticsScalarType>,
                            Plato::ScalarMultiVectorT<KinematicsScalarType>,
                            Plato::ScalarVectorT<ControlScalarType>,
                            Plato::Scalar > tExpEval;
        
        tExpEval.parse_expression(mExpression.c_str());
        tExpEval.setup_storage(tNumCells, 1);
        tExpEval.set_variable("E0", mE0);
printf("Expression: %s\n", mExpression.c_str());    
printf("E0: %lf\n", mE0);    
printf("Element Youngs Modulus Values\n");
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            tExpEval.set_variable("tElementDensity", tElementDensity, aCellOrdinal);
            tExpEval.evaluate_expression( aCellOrdinal, tElementYoungsModulusValues );
            printf("%lf\n", tElementYoungsModulusValues(aCellOrdinal,0));
        },"Compute Youngs Modulus for each Element");
        Kokkos::fence();
        tExpEval.clear_storage();

/*
FILE *fp3 = fopen("ElemE.txt", "w");
std::cout << "Element Young Modulus" << std::endl;
for(int i=0; i<tNumCells; ++i)
{
    fprintf(fp3, "%lf\n", tElementYoungsModulusValues(i,0));
    std::cout << tElementYoungsModulusValues(i,0) << std::endl;
}
fclose(fp3);
*/


        auto tPoissonsRatio = mPoissonsRatio;
        auto tNumVoigtTerms = mNumVoigtTerms;
        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
        {
            StateT tTemperature = aTemperature(aCellOrdinal);
            auto tCurYoungsModulus = tElementYoungsModulusValues(aCellOrdinal,0);
            Plato::IsotropicStiffnessConstant<EvaluationType::SpatialDim, KineticsScalarType> 
                    tStiffnessConstant(tCurYoungsModulus, tPoissonsRatio);            
            
            // compute thermal strain
            //
            StateT tstrain[tNumVoigtTerms] = {0};
            for( int iDim=0; iDim<EvaluationType::SpatialDim; iDim++ ){
                tstrain[iDim] = tScaling * tThermalExpansivityConstant(tVoigtMap.I[iDim], tVoigtMap.J[iDim])
                            * (tTemperature - tRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
                aStress(aCellOrdinal,iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<tNumVoigtTerms; jVoigt++){
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
