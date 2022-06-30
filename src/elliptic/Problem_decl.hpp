#pragma once

#include "PlatoMesh.hpp"
#include "SpatialModel.hpp"
#include "alg/ParallelComm.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "elliptic/VectorFunction.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsType>
class Problem: public Plato::AbstractProblem
{
private:

    using Criterion       = std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>;
    using Criteria        = std::map<std::string, Criterion>;

    using LinearCriterion = std::shared_ptr<Plato::Geometric::ScalarFunctionBase>;
    using LinearCriteria  = std::map<std::string, LinearCriterion>;

    using ElementType = typename PhysicsType::ElementType;
    using TopoElementType = typename ElementType::TopoElementType;

    using VectorFunctionType = Plato::Elliptic::VectorFunction<PhysicsType>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    LinearCriteria mLinearCriteria;
    Criteria       mCriteria;

    Plato::OrdinalType mNumNewtonSteps;
    Plato::Scalar      mNewtonResTol, mNewtonIncTol;

    bool mSaveState;

    Plato::ScalarMultiVector mAdjoint;
    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    bool mIsSelfAdjoint; /*!< indicates if problem is self-adjoint */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::OrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

    std::shared_ptr<Plato::MultipointConstraints> mMPCs; /*!< multipoint constraint interface */

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
     * \brief Is criterion independent of the solution state?
     * \param [in] aName Name of criterion.
    **********************************************************************************/
    bool criterionIsLinear( const std::string & aName) override;

    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    void output(const std::string & aFilepath) override;

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    void applyStateConstraints(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector & aVector,
            Plato::Scalar aScale
    );

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
    Plato::Solutions solution(const Plato::ScalarVector & aControl);

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
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion);

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
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion);

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
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

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aProblemParams input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Teuchos::ParameterList& aProblemParams);

    /***************************************************************************//**
     * \brief Set essential (Dirichlet) boundary conditions
     * \param [in] aDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aValues values associated with Dirichlet degrees of freedom
    *******************************************************************************/
    void setEssentialBoundaryConditions(const Plato::OrdinalVector & aDofs, const Plato::ScalarVector & aValues);

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams);

    void applyAdjointConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector);

    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    Plato::Solutions getSolution() const override;
};
// class Problem

} // namespace Elliptic

} // namespace Plato
