#pragma once

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "AnalyzeOutput.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"
#include "SpatialModel.hpp"

#include "ParseTools.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"
#include "PlatoUtilities.hpp"
#include "AnalyzeMacros.hpp"

#include "helmholtz/VectorFunction.hpp"

#include "solver/ParallelComm.hpp"
#include "solver/PlatoSolverFactory.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public Plato::AbstractProblem
{
private:

    using ElementType = typename PhysicsType::ElementType;

    using VectorFunctionType = Plato::Helmholtz::VectorFunction<PhysicsType>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    rcp<Plato::AbstractSolver> mSolver;

    std::string mPDEType; /*!< partial differential equation type */
    std::string mPhysics; /*!< physics used for the simulation */

public:
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    Problem(
      Plato::Mesh              aMesh,
      Teuchos::ParameterList & aProblemParams,
      Comm::Machine            aMachine
    );

    ~Problem();

    Plato::OrdinalType numNodes() const;

    Plato::OrdinalType numCells() const;
    
    Plato::OrdinalType numDofsPerCell() const;

    Plato::OrdinalType numNodesPerCell() const;

    Plato::OrdinalType numDofsPerNode() const;

    Plato::OrdinalType numControlsPerNode() const;

    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    void output(const std::string & aFilepath) override;

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution);

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return solution database
    **********************************************************************************/
    Plato::Solutions
    solution(const Plato::ScalarVector & aControl);

    /******************************************************************************//**
     * \brief Solve system of equations related to chain rule of Helmholtz filter 
     * for gradients
     * \param [in] aControl 1D view of criterion partial derivative 
     * wrt filtered control
     * \param [in] aName Name of criterion (is just a dummy for Helmhomtz to 
     * match signature of base class virtual function).
     * \return 1D view - criterion partial derivative wrt unfiltered control
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override;

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override;

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override;

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override;

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override;

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override;

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Plato::Mesh& aMesh, Teuchos::ParameterList& aProblemParams);

    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    Plato::Solutions getSolution() const;
};
// class Problem

} // namespace Helmholtz

} // namespace Plato
