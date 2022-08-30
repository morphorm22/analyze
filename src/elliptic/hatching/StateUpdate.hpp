#pragma once

#include "SmallStrain.hpp"
#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "GradientMatrix.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/hatching/EvaluationTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<typename PhysicsType>
class StateUpdate : public Plato::WorksetBase<typename PhysicsType::ElementType>
/******************************************************************************/
{
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumLocalDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumLocalStatesPerGP;
    using Plato::WorksetBase<ElementType>::mNumDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell;
    using Plato::WorksetBase<ElementType>::mNumVoigtTerms;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;

  public:
    template <typename EvaluationType>
    class ResidualFunction
    {
        using GlobalStateScalarType = typename EvaluationType::GlobalStateScalarType;
        using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
        using ResultScalarType      = typename EvaluationType::ResultScalarType;

      public:

        virtual void
        evaluate(
            const Plato::ScalarMultiVectorT<GlobalStateScalarType > & aGlobalState,
            const Plato::ScalarArray3DT    <Plato::Scalar         > & aLocalState,
            const Plato::ScalarArray3DT    <Plato::Scalar         > & aLocalStatePrev,
            const Plato::ScalarArray3DT    <ConfigScalarType      > & aConfig,
            const Plato::ScalarArray3DT    <ResultScalarType      > & aResult
        )
        {
            Plato::OrdinalType tNumCells = aGlobalState.extent(0);

            using ordT = typename Plato::ScalarVector::size_type;

            using StrainScalarType = typename Plato::fad_type_t<ElementType, GlobalStateScalarType, ConfigScalarType>;

            Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
            Plato::SmallStrain<ElementType>           tComputeVoigtStrainIncrement;

            auto tCubPoints = ElementType::getCubPoints();
            auto tCubWeights = ElementType::getCubWeights();
            auto tNumGP = tCubWeights.size();

            Kokkos::parallel_for("residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumGP}),
            KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
            {
                ConfigScalarType tVolume(0.0);

                Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

                Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrainIncrement(0.0);
                Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

                auto tCubPoint = tCubPoints(iGpOrdinal);

                tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

                tComputeVoigtStrainIncrement(iCellOrdinal, tStrainIncrement, aGlobalState, tGradient);

                for (ordT iTerm=0; iTerm<mNumVoigtTerms; iTerm++)
                {
                    aResult(iCellOrdinal, iGpOrdinal, iTerm) = - tStrainIncrement(iTerm)
                                                              + aLocalState(iCellOrdinal, iGpOrdinal, iTerm)
                                                              - aLocalStatePrev(iCellOrdinal, iGpOrdinal, iTerm);
                }
            });
        }
    };

  private:

    Plato::SpatialModel & mSpatialModel;

    using Residual  = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::Residual;
    using Jacobian  = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::Jacobian;
    using GradientC = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientC;
    using GradientX = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Hatching::Evaluation<ElementType>::GradientZ;


public:
    /******************************************************************************/
    explicit
    StateUpdate(Plato::SpatialModel & aSpatialModel) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel(aSpatialModel) {}
    /******************************************************************************/

    /******************************************************************************/
    ~StateUpdate(){}
    /******************************************************************************/

    /******************************************************************************/
    void
    operator()(
        Plato::DataMap       const & aDataMap,
        Plato::ScalarArray3D const & aPreviousStrain,
        Plato::ScalarArray3D const & aUpdatedStrain
    )
    /******************************************************************************/
    {
        using ordT = Plato::ScalarVector::size_type;

        auto tStrainInc = aDataMap.scalarArray3Ds.at("strain increment");

        Plato::OrdinalType tNumCells  = tStrainInc.extent(0);
        Plato::OrdinalType tNumGP = tStrainInc.extent(1);
        Plato::OrdinalType tNumTerms  = tStrainInc.extent(2);

        Kokkos::parallel_for("update local state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumGP}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            for (ordT iTerm=0; iTerm<tNumTerms; iTerm++)
            {
                aUpdatedStrain(iCellOrdinal, iGpOrdinal, iTerm) = aPreviousStrain(iCellOrdinal, iGpOrdinal, iTerm)
                                                                + tStrainInc(iCellOrdinal, iGpOrdinal, iTerm);
            }
        });

        if (aDataMap.scalarMultiVectors.count("total strain"))
        {
            auto tTotalStrain = aDataMap.scalarMultiVectors.at("total strain");

            Plato::OrdinalType tNumCells = tTotalStrain.extent(0);
            Plato::OrdinalType tNumTerms = tTotalStrain.extent(1);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
            {
                for (ordT iTerm=0; iTerm<tNumTerms; iTerm++)
                {
                    tTotalStrain(aCellOrdinal, iTerm) = 0.0;
                    for (ordT iGpOrdinal=0; iGpOrdinal<tNumGP; iGpOrdinal++)
                    {
                        tTotalStrain(aCellOrdinal, iTerm) += aUpdatedStrain(aCellOrdinal, iGpOrdinal, iTerm);
                    }
                    tTotalStrain(aCellOrdinal, iTerm) /= tNumGP;
                }
            }, "Save total strain");
        }
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarArray3D & aPrevLocalState
    /**************************************************************************/
    ) const
    {
        ANALYZE_THROWERR("StateUpdate::gradient_u is not implemented.");
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarArray3D & aPrevLocalState
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Jacobian::ConfigScalarType;
        using GlobalStateScalar = typename Jacobian::GlobalStateScalarType;
        using LocalStateScalar  = typename Jacobian::LocalStateScalarType;
        using ResultScalar      = typename Jacobian::ResultScalarType;

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        // create return matrix
        //
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
         Plato::CreateGlobalByLocalBlockMatrix<Plato::CrsMatrixType, ElementType>(mSpatialModel.Mesh);

        // assembly to return matrix
        //
        Plato::GlobalByLocalEntryFunctor<ElementType> tJacobianMatEntryOrdinal( tJacobianMat, mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset previous local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tPrevLocalStateWS("Previous Local State", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarArray3DT<ResultScalar> tResult("Result Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);

            ResidualFunction<Jacobian> tJacobianFunction;
            tJacobianFunction.evaluate(tGlobalStateWS, tLocalStateWS, tPrevLocalStateWS, tConfigWS, tResult);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleStateJacobianTranspose
                (mNumLocalStatesPerGP, mNumDofsPerCell, tJacobianMatEntryOrdinal, tResult, tJacobianMatEntries, tDomain);
        }
        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector  & aGlobalState,
        const Plato::ScalarArray3D & aLocalState,
        const Plato::ScalarArray3D & aPrevLocalState
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using GlobalStateScalar = typename GradientX::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientX::LocalStateScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        auto tNumCells = mSpatialModel.Mesh->NumElements();

        constexpr auto tNumGP = ElementType::mNumGaussPoints;

        // create return matrix
        //
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateGlobalByLocalBlockMatrix<Plato::CrsMatrixType, ElementType>( mSpatialModel.Mesh );

        // assembly to return matrix
        //
        Plato::GlobalByLocalEntryFunctor<ElementType>
        tJacobianMatEntryOrdinal( tJacobianMat, mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<ElementType>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tLocalStateWS("Local State", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset previous local state
            //
            Plato::ScalarArray3DT<LocalStateScalar> tPrevLocalStateWS("Previous Local State", tNumCells, tNumGP, mNumLocalStatesPerGP);
            Plato::WorksetBase<ElementType>::worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarArray3DT<ResultScalar> tResult("Result Workset", tNumCells, tNumGP, mNumLocalStatesPerGP);

            ResidualFunction<GradientX> tGradientXFunction;
            tGradientXFunction.evaluate(tGlobalStateWS, tLocalStateWS, tPrevLocalStateWS, tConfigWS, tResult);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<ElementType>::assembleStateJacobianTranspose
                (mNumLocalDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tResult, tJacobianMatEntries, tDomain);
        }
        return tJacobianMat;
    }

};

} // namespace Plato
