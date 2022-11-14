#pragma once

#include "Solutions.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************//**
 * @brief Abstract vector function (i.e. PDE) interface
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 **********************************************************************************/
template<typename EvaluationType>
class AbstractVectorFunction
{
protected:
    const Plato::SpatialDomain     & mSpatialDomain;  /*!< Plato spatial model containing mesh, meshsets, etc */
          Plato::DataMap           & mDataMap;        /*!< Plato Analyze database */
          std::vector<std::string>   mDofNames;       /*!< state dof names */

public:

    using AbstractType = Plato::Elliptic::Hatching::AbstractVectorFunction<EvaluationType>;

    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aSpatialDomain Plato spatial model
     * @param [in] aDataMap Plato Analyze database
    **********************************************************************************/
    explicit
    AbstractVectorFunction(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        mSpatialDomain (aSpatialDomain),
        mDataMap      (aDataMap)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    virtual ~AbstractVectorFunction()
    {
    }

    /****************************************************************************//**
    * @brief Return reference to mesh database
    * @return volume mesh database
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
     * @brief Evaluate vector function
     * @param [in] aState 2D array with global state variables (C,DOF)
     * @param [in] aState 2D array with local state variables (C,GP*LS)
     * @param [in] aControl 2D array with control variables (C,N)
     * @param [in] aConfig 3D array with control variables (C,N,D)
     * @param [in] aResult 1D array with control variables (C,DOF)
     * @param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * GP = number of gauss points per cell, LS = number of local states
    **********************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT <typename EvaluationType::GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <typename EvaluationType::LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <typename EvaluationType::ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Evaluate vector function
     * @param [in] aState 2D array with global state variables (C,DOF)
     * @param [in] aState 2D array with local state variables (C,GP*LS)
     * @param [in] aControl 2D array with control variables (C,N)
     * @param [in] aConfig 3D array with control variables (C,N,D)
     * @param [in] aResult 1D array with control variables (C,DOF)
     * @param [in] aTimeStep current time step
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * GP = number of gauss points per cell, LS = number of local states
    **********************************************************************************/
    virtual void
    evaluate_boundary(
        const Plato::SpatialModel                                                        & aModel,
        const Plato::ScalarMultiVectorT <typename EvaluationType::GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarArray3DT     <typename EvaluationType::LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <typename EvaluationType::ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <typename EvaluationType::ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <typename EvaluationType::ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0) const = 0;
};
// class AbstractVectorFunction

} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
