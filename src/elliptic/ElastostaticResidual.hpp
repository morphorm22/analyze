#ifndef ELASTOSTATIC_RESIDUAL_HPP
#define ELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "PlatoTypes.hpp"
#include "FadTypes.hpp"
#include "SmallStrain.hpp"
#include "LinearStress.hpp"
#include "GeneralStressDivergence.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "ApplyWeighting.hpp"
#include "CellForcing.hpp"
#include "ToMap.hpp"
#include "VonMisesYieldFunction.hpp"

#include "ElasticModelFactory.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"

#include "ExpInstMacros.hpp"

// includes below here are verified to be necessary
#include "MechanicsElement.hpp"
#include "GradientMatrix.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Elastostatic vector function interface
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * \tparam IndicatorFunctionType penalty function used for density-based methods
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual : 
    public EvaluationType::ElementType,
    public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;
    Plato::CellForcing<ElementType> mCellForcing;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType>> mBoundaryLoads;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

    std::vector<std::string> mPlotTable;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze database
     * \param [in] aProblemParams input parameters for overall problem
     * \param [in] aPenaltyParams input parameters for penalty function
    **********************************************************************************/
    ElastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction),
        mBodyLoads         (nullptr),
        mBoundaryLoads     (nullptr)
    {
        // obligatory: define dof names in order
        mDofNames.push_back("displacement X");
        if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
        if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");

        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType>>(aProblemParams.sublist("Natural Boundary Conditions"));
        }
        // parse cell problem forcing
        //
        if(aProblemParams.isSublist("Cell Problem Forcing"))
        {
            Plato::OrdinalType tColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
            mCellForcing.setCellStiffness(mMaterialModel->getStiffnessMatrix());
            mCellForcing.setColumnIndex(tColumnIndex);
        }

        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
          mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }

    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    {
      // No scaling, addition, or removal of data necessary for this physics.
      return aSolutions;
    }

    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aState 2D array with state variables (C,DOF)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
      using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType>   computeGradient;
      Plato::SmallStrain<ElementType>             computeVoigtStrain;
      Plato::GeneralStressDivergence<ElementType> computeStressDivergence;

      Plato::LinearStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      auto& applyWeighting = mApplyWeighting;
      auto& cellForcing = mCellForcing;
    
      Kokkos::parallel_for("compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          ConfigScalarType tVolume(0.0);

          Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

          Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
          Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

          auto tCubPoint = tCubPoints(iGpOrdinal);

          computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

          computeVoigtStrain(iCellOrdinal, tStrain, aState, tGradient);

          computeVoigtStress(tStress, tStrain);

          cellForcing(tStress);

          tVolume *= tCubWeights(iGpOrdinal);

          auto tBasisValues = ElementType::basisValues(tCubPoint);
          applyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

          computeStressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellStrain(iCellOrdinal,i), tVolume*tStrain(i));
              Kokkos::atomic_add(&tCellStress(iCellOrdinal,i), tVolume*tStress(i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      LAMBDA_EXPRESSION(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellStrain(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

      if(std::count(mPlotTable.begin(), mPlotTable.end(), "strain")) { Plato::toMap(mDataMap, tCellStrain, "strain", mSpatialDomain); }
      if(std::count(mPlotTable.begin(), mPlotTable.end(), "stress")) { Plato::toMap(mDataMap, tCellStress, "stress", mSpatialDomain); }
      if(std::count(mPlotTable.begin(), mPlotTable.end(), "Vonmises")) { this->outputVonMises(tCellStress, mSpatialDomain); }
    }
    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aState 2D array with state variables (C,DOF)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
        }
    }


    /**********************************************************************//**
     * \brief Compute Von Mises stress field and copy data into output data map
     * \param [in] aCauchyStress Cauchy stress tensor
    **************************************************************************/
    void
    outputVonMises(
        const Plato::ScalarMultiVectorT<ResultScalarType> & aCauchyStress,
        const Plato::SpatialDomain                        & aSpatialDomain
    ) const
    {
            auto tNumCells = aSpatialDomain.numCells();
            Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;
            Plato::ScalarVectorT<ResultScalarType> tVonMises("Von Mises", tNumCells);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                ResultScalarType tCellVonMises(0);
                tComputeVonMises(aCellOrdinal, aCauchyStress, tCellVonMises);
                tVonMises(aCellOrdinal) = tCellVonMises;
            }, "Compute VonMises Stress");

            Plato::toMap(mDataMap, tVonMises, "Vonmises", aSpatialDomain);
    }
};
// class ElastostaticResidual

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
//PLATO_EXPL_DEC(Plato::Elliptic::ElastostaticResidual, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
//PLATO_EXPL_DEC(Plato::Elliptic::ElastostaticResidual, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
//PLATO_EXPL_DEC(Plato::Elliptic::ElastostaticResidual, Plato::MechanicsElement<Plato::Tet10>, 3)
#endif

#endif
