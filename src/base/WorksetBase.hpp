#ifndef WORKSET_BASE_HPP
#define WORKSET_BASE_HPP

#include <cassert>

#include "ImplicitFunctors.hpp"
#include "AnalyzeMacros.hpp"
#include "FadTypes.hpp"
#include "BLAS1.hpp"

#include "Assembly.hpp"


namespace Plato
{

/******************************************************************************/
/*! Base class for workset functionality.
*/
/******************************************************************************/
template<typename ElementType>
class WorksetBase : public ElementType
{
protected:
    Plato::OrdinalType mNumCells; /*!< local number of elements */
    Plato::OrdinalType mNumNodes; /*!< local number of nodes */

    using ElementType::mNumDofsPerNode;      /*!< number of degrees of freedom per node */
    using ElementType::mNumControl;          /*!< number of control vectors, i.e. materials */
    using ElementType::mNumNodesPerCell;     /*!< number of nodes per element */
    using ElementType::mNumDofsPerCell;      /*!< number of global degrees of freedom, e.g. displacements, per element  */
    using ElementType::mNumLocalDofsPerCell; /*!< number of local degrees of freedom, e.g. plasticity variables, per element  */
    using ElementType::mNumNodeStatePerNode; /*!< number of pressure states per node  */

    using StateFad      = typename Plato::FadTypes<ElementType>::StateFad;          /*!< global state AD type */
    using LocalStateFad = typename Plato::FadTypes<ElementType>::LocalStateFad;     /*!< local state AD type */
    using NodeStateFad  = typename Plato::FadTypes<ElementType>::NodeStateFad;      /*!< node state AD type */
    using ControlFad    = typename Plato::FadTypes<ElementType>::ControlFad;        /*!< control AD type */
    using ConfigFad     = typename Plato::FadTypes<ElementType>::ConfigFad;         /*!< configuration AD type */

    /*!< number of spatial dimensions */
    static constexpr Plato::OrdinalType mSpaceDim = ElementType::mNumSpatialDims;          
    /*!< number of configuration degrees of freedom per element  */
    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mSpaceDim * mNumNodesPerCell; 

public:
    /*!< local-to-global ID map for global state */
    Plato::VectorEntryOrdinal<mSpaceDim, mNumDofsPerNode,      mNumNodesPerCell> mGlobalStateEntryOrdinal; 
    /*!< local-to-global ID map for node state */
    Plato::VectorEntryOrdinal<mSpaceDim, mNumNodeStatePerNode, mNumNodesPerCell> mNodeStateEntryOrdinal;   
    /*!< local-to-global ID map for control */
    Plato::VectorEntryOrdinal<mSpaceDim, mNumControl,          mNumNodesPerCell> mControlEntryOrdinal;     
    /*!< local-to-global ID map for configuration */
    Plato::VectorEntryOrdinal<mSpaceDim, mSpaceDim,            mNumNodesPerCell> mConfigEntryOrdinal;      
    /*!< node coordinates database */
    Plato::NodeCoordinate<mSpaceDim, mNumNodesPerCell> mNodeCoordinate; 

public:
    /******************************************************************************//**
     * \brief Return number of cells
     * \return number of cells
    **********************************************************************************/
    decltype(mNumCells) numCells() const
    {
        return (mNumCells);
    }

    /******************************************************************************//**
     * \brief Return number of nodes
     * \return number of nodes
    **********************************************************************************/
    decltype(mNumNodes) numNodes() const
    {
        return (mNumNodes);
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
    **********************************************************************************/
    WorksetBase(Plato::Mesh aMesh) :
            mNumCells(aMesh->NumElements()),
            mNumNodes(aMesh->NumNodes()),
            mGlobalStateEntryOrdinal(Plato::VectorEntryOrdinal<mSpaceDim, mNumDofsPerNode, mNumNodesPerCell>(aMesh)),
            mNodeStateEntryOrdinal(Plato::VectorEntryOrdinal<mSpaceDim, mNumNodeStatePerNode, mNumNodesPerCell>(aMesh)),
            mControlEntryOrdinal(Plato::VectorEntryOrdinal<mSpaceDim, mNumControl, mNumNodesPerCell>(aMesh)),
            mConfigEntryOrdinal(Plato::VectorEntryOrdinal<mSpaceDim, mSpaceDim, mNumNodesPerCell>(aMesh)),
            mNodeCoordinate(Plato::NodeCoordinate<mSpaceDim, mNumNodesPerCell>(aMesh))
    {
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (scalar type), as a 2-D Kokkos::View
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <Plato::Scalar> & aControlWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        if(aDomain.isFixedBlock())
        {
            Plato::ScalarVector tFixedControl("fixed control", aControl.size());
            Plato::blas1::fill(1.0, tFixedControl);
            Plato::workset_control_scalar_scalar<mNumNodesPerCell>(
                aDomain, mControlEntryOrdinal, tFixedControl, aControlWS);
        }
        else
        {
            Plato::workset_control_scalar_scalar<mNumNodesPerCell>(
                aDomain, mControlEntryOrdinal, aControl, aControlWS);
        }
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (scalar type), as a 2-D Kokkos::View
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <Plato::Scalar> & aControlWS
    ) const
    {
        Plato::workset_control_scalar_scalar<mNumNodesPerCell>(
            mNumCells, mControlEntryOrdinal, aControl, aControlWS);
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (AD type), as a 2-D Kokkos::View
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <ControlFad>    & aFadControlWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        if(aDomain.isFixedBlock())
        {
            Plato::ScalarVector tFixedControl("fixed control", aControl.size());
            Plato::blas1::fill(1.0, tFixedControl);
            Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlFad>(
                aDomain, mControlEntryOrdinal, tFixedControl, aFadControlWS);
        }
        else
        {
            Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlFad>(
                aDomain, mControlEntryOrdinal, aControl, aFadControlWS);
        }
    }

    /******************************************************************************//**
     * \brief Get controls workset, e.g. design/optimization variables
     * \param [in] aControl controls (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadControlWS controls workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetControl(
        const Plato::ScalarVectorT      <Plato::Scalar> & aControl,
              Plato::ScalarMultiVectorT <ControlFad>    & aFadControlWS
    ) const
    {
        Plato::workset_control_scalar_fad<mNumNodesPerCell, ControlFad>(
            mNumCells, mControlEntryOrdinal, aControl, aFadControlWS);
    }

    /******************************************************************************//**
     * \brief Get global state workset, e.g. displacements in mechanic problem
     * \param [in] aState global state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadStateWS global state workset (scalar type), as a 2-D Kokkos::View
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    void
    worksetState(
        const Plato::ScalarVectorT      <Plato::Scalar> & aState,
              Plato::ScalarMultiVectorT <Plato::Scalar> & aStateWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        Plato::workset_state_scalar_scalar<mNumDofsPerNode, mNumNodesPerCell>(
            aDomain, mGlobalStateEntryOrdinal, aState, aStateWS);
    }

    /******************************************************************************//**
     * \brief Get global state workset, e.g. displacements in mechanic problem
     * \param [in] aState global state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadStateWS global state workset (scalar type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetState(
        const Plato::ScalarVectorT      <Plato::Scalar> & aState,
              Plato::ScalarMultiVectorT <Plato::Scalar> & aStateWS
    ) const
    {
        Plato::workset_state_scalar_scalar<mNumDofsPerNode, mNumNodesPerCell>(
            mNumCells, mGlobalStateEntryOrdinal, aState, aStateWS);
    }

    /******************************************************************************//**
     * \brief Get global state workset, e.g. displacements in mechanic problem
     * \param [in] aState global state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadStateWS global state workset (AD type), as a 2-D Kokkos::View
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    void
    worksetState(
        const Plato::ScalarVectorT      <Plato::Scalar> & aState,
              Plato::ScalarMultiVectorT <StateFad>      & aFadStateWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
        Plato::workset_state_scalar_fad<mNumDofsPerNode, mNumNodesPerCell, StateFad>(
            aDomain, mGlobalStateEntryOrdinal, aState, aFadStateWS);
    }

    /******************************************************************************//**
     * \brief Get global state workset, e.g. displacements in mechanic problem
     * \param [in] aState global state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadStateWS global state workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetState(
        const Plato::ScalarVectorT      <Plato::Scalar> & aState,
              Plato::ScalarMultiVectorT <StateFad>      & aFadStateWS
    ) const
    {
        Plato::workset_state_scalar_fad<mNumDofsPerNode, mNumNodesPerCell, StateFad>(
            mNumCells, mGlobalStateEntryOrdinal, aState, aFadStateWS);
    }

    /******************************************************************************//**
     * \brief Get local state workset, e.g. history variables in plasticity problems
     * \param [in] aLocalState local state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aLocalStateWS local state workset (scalar type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetLocalState(
        const Plato::ScalarArray3DT<Plato::Scalar> & aLocalState,
              Plato::ScalarArray3DT<Plato::Scalar> & aLocalStateWS
    ) const
    {
      Plato::workset_local_state_scalar_scalar<mNumLocalDofsPerCell>(
              mNumCells, aLocalState, aLocalStateWS);
    }

    /******************************************************************************//**
     * \brief Get local state workset, e.g. history variables in plasticity problems
     * \param [in] aLocalState local state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aLocalStateWS local state workset (scalar type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetLocalState(
        const Plato::ScalarArray3DT<Plato::Scalar> & aLocalState,
              Plato::ScalarArray3DT<Plato::Scalar> & aLocalStateWS,
        const Plato::SpatialDomain                 & aDomain
    ) const
    {
      Plato::workset_local_state_scalar_scalar<mNumLocalDofsPerCell>(
              aDomain, aLocalState, aLocalStateWS);
    }

    /******************************************************************************//**
     * \brief Get local state workset, e.g. history variables in plasticity problems
     * \param [in] aLocalState local state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadLocalStateWS local state workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetLocalState(
        const Plato::ScalarArray3DT<Plato::Scalar> & aLocalState,
              Plato::ScalarArray3DT<LocalStateFad> & aFadLocalStateWS
    ) const
    {
      Plato::workset_local_state_scalar_fad<mNumLocalDofsPerCell, LocalStateFad>(
              mNumCells, aLocalState, aFadLocalStateWS);
    }

    /******************************************************************************//**
     * \brief Get local state workset, e.g. history variables in plasticity problems
     * \param [in] aLocalState local state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadLocalStateWS local state workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetLocalState(
        const Plato::ScalarArray3DT<Plato::Scalar> & aLocalState,
              Plato::ScalarArray3DT<LocalStateFad> & aFadLocalStateWS,
        const Plato::SpatialDomain                 & aDomain
    ) const
    {
      Plato::workset_local_state_scalar_fad<mNumLocalDofsPerCell, LocalStateFad>(
              aDomain, aLocalState, aFadLocalStateWS);
    }

    /******************************************************************************//**
     * \brief Get node state workset, e.g. projected pressure gradient in stabilized
     *        mechanics problem for each cell
     * \param [in] aState node state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aNodeStateWS node state workset (scalar type), as a 2-D Kokkos::View
    **********************************************************************************/
    void 
    worksetNodeState( 
      const ScalarVectorT            <Plato::Scalar> & aState,
            Plato::ScalarMultiVectorT<Plato::Scalar> & aNodeStateWS 
    ) const
    {
      Plato::workset_state_scalar_scalar<mNumNodeStatePerNode, mNumNodesPerCell>(
        mNumCells, mNodeStateEntryOrdinal, aState, aNodeStateWS
      );
    }

    /******************************************************************************//**
     * \brief Get node state workset, e.g. projected pressure gradient in stabilized
     *        mechanics problem for each cell
     * \param [in] aState node state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aNodeStateWS node state workset (scalar type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetNodeState(
      const Plato::ScalarVectorT      <Plato::Scalar> & aState,
            Plato::ScalarMultiVectorT <Plato::Scalar> & aNodeStateWS,
      const Plato::SpatialDomain                      & aDomain
    ) const
    {
      Plato::workset_state_scalar_scalar<mNumNodeStatePerNode, mNumNodesPerCell>(
              aDomain, mNodeStateEntryOrdinal, aState, aNodeStateWS);
    }

    /******************************************************************************//**
     * \brief Get node state workset, e.g. projected pressure gradient in stabilized
     *        mechanics problem for each cell
     * \param [in] aState node state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadStateWS node state workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void 
    worksetNodeState( 
      const Plato::ScalarVectorT      <Plato::Scalar> & aState,
            Plato::ScalarMultiVectorT <NodeStateFad>  & aFadStateWS 
    ) const
    {
      Plato::workset_state_scalar_fad<mNumNodeStatePerNode, mNumNodesPerCell, NodeStateFad>(
        mNumCells, mNodeStateEntryOrdinal, aState, aFadStateWS
      );
    }

    /******************************************************************************//**
     * \brief Get node state workset, e.g. projected pressure gradient in stabilized
     *        mechanics problem for each cell
     * \param [in] aState node state (scalar type), as a 1-D Kokkos::View
     * \param [in/out] aFadStateWS node state workset (AD type), as a 2-D Kokkos::View
    **********************************************************************************/
    void
    worksetNodeState(
        const Plato::ScalarVectorT      <Plato::Scalar> & aState,
              Plato::ScalarMultiVectorT <NodeStateFad>  & aFadStateWS,
        const Plato::SpatialDomain                      & aDomain
    ) const
    {
      Plato::workset_state_scalar_fad<mNumNodeStatePerNode, mNumNodesPerCell, NodeStateFad>(
        aDomain, mNodeStateEntryOrdinal, aState, aFadStateWS
      );
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aConfigWS configuration workset (scalar type), as a 3-D Kokkos::View
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    void
    worksetConfig(
              Plato::ScalarArray3DT <Plato::Scalar> & aConfigWS,
        const Plato::SpatialDomain                  & aDomain
    ) const
    {
      Plato::workset_config_scalar<mSpaceDim, mNumNodesPerCell>(
          aDomain, mNodeCoordinate, aConfigWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aConfigWS configuration workset (scalar type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
        Plato::ScalarArray3DT <Plato::Scalar> & aConfigWS
    ) const
    {
      Plato::workset_config_scalar<mSpaceDim, mNumNodesPerCell>(
          mNumCells, mNodeCoordinate, aConfigWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aReturnValue configuration workset (AD type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
              Plato::ScalarArray3DT <ConfigFad> & aFadConfigWS,
        const Plato::SpatialDomain              & aDomain
    ) const
    {
      Plato::workset_config_fad<mSpaceDim, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigFad>(
          aDomain, mNodeCoordinate, aFadConfigWS);
    }

    /******************************************************************************//**
     * \brief Get configuration workset, i.e. coordinates for each cell
     * \param [in/out] aReturnValue configuration workset (AD type), as a 3-D Kokkos::View
    **********************************************************************************/
    void
    worksetConfig(
        Plato::ScalarArray3DT <ConfigFad> & aFadConfigWS
    ) const
    {
      Plato::workset_config_fad<mSpaceDim, mNumNodesPerCell, mNumConfigDofsPerCell, ConfigFad>(
          mNumCells, mNodeCoordinate, aFadConfigWS);
    }

    /******************************************************************************//**
     * \brief Assemble residual vector
     *
     * \tparam ResidualWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledResidualType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset
     * \param [in/out] aReturnValue assembled residual
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    template<class ResidualWorksetType, class AssembledResidualType>
    void assembleResidual(
        const ResidualWorksetType   & aResidualWorkset,
              AssembledResidualType & aReturnValue,
        const Plato::SpatialDomain  & aDomain
    ) const
    {
        Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
            (aDomain, WorksetBase<ElementType>::mGlobalStateEntryOrdinal, aResidualWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble residual vector
     *
     * \tparam ResidualWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledResidualType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset
     * \param [in/out] aReturnValue assembled residual
    **********************************************************************************/
    template<class ResidualWorksetType, class AssembledResidualType>
    void assembleResidual(
        const ResidualWorksetType   & aResidualWorkset,
              AssembledResidualType & aReturnValue
    ) const
    {
        Plato::assemble_residual<mNumNodesPerCell, mNumDofsPerNode>
            (mNumCells, WorksetBase<ElementType>::mGlobalStateEntryOrdinal, aResidualWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to global states (U)
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleVectorGradientU(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_vector_gradient<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mGlobalStateEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to global states (U) - specialized
     * for automatic differentiation types
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType     Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, 
             class OutType>
    void assembleVectorGradientFadU(
        const WorksetType & aWorkset, 
        OutType & aOutput
    ) const
    {
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(
            mNumCells, mGlobalStateEntryOrdinal, aWorkset, aOutput);
    }

    /// @brief assemble partial derivative with respect to state
    /// @tparam WorksetType workset typename 
    /// @tparam OutType     output typename
    /// @param aDomain  contains mesh and model information
    /// @param aWorkset workset data
    /// @param aOutput  assembled partial derivative
    template<class WorksetType, 
             class OutType>
    void assembleVectorGradientFadU(
        const Plato::SpatialDomain & aDomain,
        const WorksetType          & aWorkset, 
              OutType              & aOutput
    ) const
    {
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(
            aDomain, mGlobalStateEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to local states (C)
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleVectorGradientC(const WorksetType & aWorkset, OutType & aOutput) const
    {
        if(mNumLocalDofsPerCell <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("Number of local degrees of freedom is set to zero. Local state variables are not defined for this application.");
        }
        Plato::flatten_vector_workset<mNumLocalDofsPerCell>(mNumCells, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to configuration (X)
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleVectorGradientX(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_vector_gradient<mNumNodesPerCell, mSpaceDim>(mNumCells, mConfigEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to configuration (X) - specialized
     * for automatic differentiation types
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType     Output container, as a 1-D Kokkos::View
     *
     * \param [in]     aResidualWorkset residual cell workset - Scalar type
     * \param [in,out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType,
             class OutType>
    void assembleVectorGradientFadX(
      const WorksetType & aWorkset, 
            OutType     & aOutput
    ) const
    {
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mSpaceDim>(
            mNumCells, mConfigEntryOrdinal, aWorkset, aOutput);
    }

    /// @brief assemble partial derivative with respect to configuration
    /// @tparam WorksetType workset typename 
    /// @tparam OutType     output typename
    /// @param aDomain  contains mesh and model information
    /// @param aWorkset workset data
    /// @param aOutput  assembled partial derivative
    template<class WorksetType, 
             class OutType>
    void assembleVectorGradientFadX(
      const Plato::SpatialDomain & aDomain,
      const WorksetType          & aWorkset,
            OutType              & aOutput
    ) const
    {
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mSpaceDim>(
            aDomain, mConfigEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to controls (Z)
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleScalarGradientZ(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_scalar_gradient<mNumNodesPerCell>(
            mNumCells, mControlEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble partial derivative with respect to controls (Z) - specialized
     * for automatic differentiation types
     *
     * \tparam WorksetType Input container, as a 2-D Kokkos::View
     * \tparam OutType     Output container, as a 1-D Kokkos::View
     *
     * \param [in] aResidualWorkset residual cell workset - Scalar type
     * \param [in/out] aReturnValue assembled residual - Scalar type
    **********************************************************************************/
    template<class WorksetType, class OutType>
    void assembleScalarGradientFadZ(const WorksetType & aWorkset, OutType & aOutput) const
    {
        Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>(
            mNumCells, mControlEntryOrdinal, aWorkset, aOutput);
    }

    /// @fn assembleScalarGradientFadZ
    /// @brief assemble forward automatically differentiated (FAD) partial derivative with respect to controls
    /// @tparam WorksetType workset typename
    /// @tparam OutType     output typename
    /// @param [in]     aDomain  contains mesh and model information
    /// @param [in]     aWorkset worset data
    /// @param [in,out] aOutput  assembled partial derivative
    template<class WorksetType, 
             class OutType>
    void assembleScalarGradientFadZ(
      const Plato::SpatialDomain & aDomain,
      const WorksetType          & aWorkset,
            OutType              & aOutput
    ) const
    {
      Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>(
        aDomain, mControlEntryOrdinal, aWorkset, aOutput);
    }

    /******************************************************************************//**
     * \brief Assemble Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRows number of rows
     * \param [in] aNumColumns number of columns
     * \param [in] aMatrixEntryOrdinal container of Jacobian entry ordinal (local-to-global ID map)
     * \param [in] aJacobianWorkset Jacobian cell workset
     * \param [in/out] aReturnValue assembled Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void
    assembleJacobianFad(
              Plato::OrdinalType         aNumRows,
              Plato::OrdinalType         aNumColumns,
        const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
        const JacobianWorksetType      & aJacobianWorkset,
              AssembledJacobianType    & aReturnValue
    ) const
    {
      Plato::assemble_jacobian_fad(
        mNumCells, aNumRows, aNumColumns, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue
      );
    }

    /******************************************************************************//**
     * \brief Assemble Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRows number of rows
     * \param [in] aNumColumns number of columns
     * \param [in] aMatrixEntryOrdinal container of Jacobian entry ordinal (local-to-global ID map)
     * \param [in] aJacobianWorkset Jacobian cell workset
     * \param [in/out] aReturnValue assembled Jacobian
     * \param [in] aDomain Domain containing elements to be added to workset
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void
    assembleJacobianFad(
              Plato::OrdinalType         aNumRows,
              Plato::OrdinalType         aNumColumns,
        const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
        const JacobianWorksetType      & aJacobianWorkset,
              AssembledJacobianType    & aReturnValue,
        const Plato::SpatialDomain     & aDomain
    ) const
    {
        Plato::assemble_jacobian_fad(aDomain, aNumRows, aNumColumns, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRowsPerCell     number of rows per matrix
     * \param [in] aNumColumnsPerCell  number of columns per matrix
     * \param [in] aMatrixEntryOrdinal Jacobian entry ordinal (i.e. local-to-global ID map)
     * \param [in] aJacobianWorkset    workset of cell Jacobians
     * \param [in/out] aReturnValue    assembled transposed Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class AssembledJacobianType>
    void assembleJacobian(Plato::OrdinalType aNumRowsPerCell,
                          Plato::OrdinalType aNumColumnsPerCell,
                          const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
                          const Plato::ScalarArray3D & aJacobianWorkset,
                          AssembledJacobianType & aReturnValue) const
    {
        Plato::assemble_jacobian(mNumCells, aNumRowsPerCell, aNumColumnsPerCell, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble transpose Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRowsPerCell     number of rows per matrix - (use row count from untransposed matrix)
     * \param [in] aNumColumnsPerCell  number of columns per matrix - (use column count from untransposed matrix)
     * \param [in] aMatrixEntryOrdinal Jacobian entry ordinal (i.e. local-to-global ID map)
     * \param [in] aJacobianWorkset    workset of cell Jacobians
     * \param [in/out] aReturnValue    assembled transposed Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void
    assembleTransposeJacobian(
              Plato::OrdinalType         aNumRowsPerCell,
              Plato::OrdinalType         aNumColumnsPerCell,
        const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
        const JacobianWorksetType      & aJacobianWorkset,
              AssembledJacobianType    & aReturnValue,
        const Plato::SpatialDomain     & aDomain
    ) const
    {
        Plato::assemble_transpose_jacobian
            (aDomain, aNumRowsPerCell, aNumColumnsPerCell, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble transpose Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRowsPerCell     number of rows per matrix - (use row count from untransposed matrix)
     * \param [in] aNumColumnsPerCell  number of columns per matrix - (use column count from untransposed matrix)
     * \param [in] aMatrixEntryOrdinal Jacobian entry ordinal (i.e. local-to-global ID map)
     * \param [in] aJacobianWorkset    workset of cell Jacobians
     * \param [in/out] aReturnValue    assembled transposed Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void
    assembleTransposeJacobian(
              Plato::OrdinalType         aNumRowsPerCell,
              Plato::OrdinalType         aNumColumnsPerCell,
        const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
        const JacobianWorksetType      & aJacobianWorkset,
              AssembledJacobianType    & aReturnValue
    ) const
    {
        Plato::assemble_transpose_jacobian
            (mNumCells, aNumRowsPerCell, aNumColumnsPerCell, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble transpose Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRowsPerCell     number of rows per matrix - (use row count from untransposed matrix)
     * \param [in] aNumColumnsPerCell  number of columns per matrix - (use column count from untransposed matrix)
     * \param [in] aMatrixEntryOrdinal Jacobian entry ordinal (i.e. local-to-global ID map)
     * \param [in] aJacobianWorkset    workset of cell Jacobians
     * \param [in/out] aReturnValue    assembled transposed Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void
    assembleStateJacobianTranspose(
              Plato::OrdinalType         aNumRowsPerCell,
              Plato::OrdinalType         aNumColumnsPerCell,
        const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
        const JacobianWorksetType      & aJacobianWorkset,
              AssembledJacobianType    & aReturnValue,
        const Plato::SpatialDomain     & aDomain
    ) const
    {
        Plato::assemble_state_jacobian_transpose
            (aDomain, aNumRowsPerCell, aNumColumnsPerCell, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

    /******************************************************************************//**
     * \brief Assemble transpose Jacobian
     *
     * \tparam MatrixEntriesOrdinalType Input container of matrix ordinal
     * \tparam JacobianWorksetType Input container, as a 2-D Kokkos::View
     * \tparam AssembledJacobianType Output container, as a 1-D Kokkos::View
     *
     * \param [in] aNumRowsPerCell     number of rows per matrix - (use row count from untransposed matrix)
     * \param [in] aNumColumnsPerCell  number of columns per matrix - (use column count from untransposed matrix)
     * \param [in] aMatrixEntryOrdinal Jacobian entry ordinal (i.e. local-to-global ID map)
     * \param [in] aJacobianWorkset    workset of cell Jacobians
     * \param [in/out] aReturnValue    assembled transposed Jacobian
    **********************************************************************************/
    template<class MatrixEntriesOrdinalType, class JacobianWorksetType, class AssembledJacobianType>
    void
    assembleStateJacobianTranspose(
              Plato::OrdinalType         aNumRowsPerCell,
              Plato::OrdinalType         aNumColumnsPerCell,
        const MatrixEntriesOrdinalType & aMatrixEntryOrdinal,
        const JacobianWorksetType      & aJacobianWorkset,
              AssembledJacobianType    & aReturnValue
    ) const
    {
        Plato::assemble_state_jacobian_transpose
            (mNumCells, aNumRowsPerCell, aNumColumnsPerCell, aMatrixEntryOrdinal, aJacobianWorkset, aReturnValue);
    }

};
// class WorksetBase

}//namespace Plato

#endif
