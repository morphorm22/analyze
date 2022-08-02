#pragma once

#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"
#include "Solutions.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************//**
 * \brief Abstract vector function (i.e. PDE) interface for Variational Multi-Scale
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
{
protected:
    const Plato::SpatialDomain     & mSpatialDomain; /*!< Plato Analyze spatial model */
          Plato::DataMap           & mDataMap;       /*!< Plato Analyze database */
          std::vector<std::string>   mDofNames;      /*!< state dof names */

public:

    using AbstractType = Plato::Stabilized::AbstractVectorFunction<EvaluationType>;

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze database
    **********************************************************************************/
    explicit
    AbstractVectorFunction(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap& aDataMap
    ) :
        mSpatialDomain (aSpatialDomain),
        mDataMap       (aDataMap)
    {
    }

    /******************************************************************************//**
     * \brief Destructor
    **********************************************************************************/
    virtual ~AbstractVectorFunction()
    {
    }

    /****************************************************************************//**
    * \brief Return reference to mesh database
    * \return volume mesh database
    ********************************************************************************/
    decltype(mSpatialDomain.Mesh) getMesh() const
    {
        return (mSpatialDomain.Mesh);
    }

    /****************************************************************************//**
    * \brief Return reference to dof names
    * \return mDofNames
    ********************************************************************************/
    const decltype(mDofNames)& getDofNames() const
    {
        return mDofNames;
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    virtual Plato::Solutions 
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const = 0;

    /******************************************************************************//**
     * \brief Evaluate vector function
     * \param [in] aState 2D array with state variables (C,DOF)
     * \param [in] aNodeState 2D array with state variables (C,D*N)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>     & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::NodeStateScalarType> & aNodeState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>   & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>    & aConfig,
              Plato::ScalarMultiVectorT <typename EvaluationType::ResultScalarType>    & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Evaluate vector function
     * \param [in] aState 2D array with state variables (C,DOF)
     * \param [in] aNodeState 2D array with state variables (C,D*N)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                      & aModel,
        const Plato::ScalarMultiVectorT <typename EvaluationType::StateScalarType>     & aState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::NodeStateScalarType> & aNodeState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>   & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>    & aConfig,
              Plato::ScalarMultiVectorT <typename EvaluationType::ResultScalarType>    & aResult,
              Plato::Scalar aTimeStep = 0.0) const {}

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aControl     control variables, e.g. design variables
     * \param [in] aTimeStep    pseudo time step
    **********************************************************************************/
    virtual void
    updateProblem(
        const Plato::ScalarMultiVector & aState,
        const Plato::ScalarVector      & aControl,
              Plato::Scalar              aTimeStep = 0.0)
    { return; }
};
// class AbstractVectorFunction

} // namespace Stabilized
} // namespace Plato
