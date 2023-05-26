#pragma once

#include "Solutions.hpp"
#include "PlatoMesh.hpp"
#include "SpatialModel.hpp"
#include "ComputedField.hpp"
#include "PlatoAbstractProblem.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "parabolic/VectorFunction.hpp"
#include "parabolic/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "parabolic/TrapezoidIntegrator.hpp"

namespace Plato
{

namespace Parabolic
{

    template<typename PhysicsType>
    class Problem: public Plato::AbstractProblem
    {
      private:

        using Criterion = std::shared_ptr<Plato::Parabolic::ScalarFunctionBase>;
        using Criteria  = std::map<std::string, Criterion>;

        using LinearCriterion = std::shared_ptr<Plato::Geometric::ScalarFunctionBase>;
        using LinearCriteria  = std::map<std::string, LinearCriterion>;

        using ElementType = typename PhysicsType::ElementType;
        using TopoElementType = typename ElementType::TopoElementType;

        using VectorFunctionType = Plato::Parabolic::VectorFunction<PhysicsType>;

        Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

        VectorFunctionType mPDEConstraint;

        Plato::Parabolic::TrapezoidIntegrator mTrapezoidIntegrator;

        Plato::OrdinalType mNumSteps, mNumNewtonSteps;
        Plato::Scalar      mTimeStep, mNewtonResTol, mNewtonIncTol;

        bool mSaveState;

        Criteria mCriteria;
        LinearCriteria mLinearCriteria;

        Plato::ScalarVector mResidual;
        Plato::ScalarVector mResidualV;

        Plato::ScalarMultiVector mAdjoints_U;
        Plato::ScalarMultiVector mAdjoints_V;

        Plato::ScalarMultiVector mState;
        Plato::ScalarMultiVector mStateDot;

        Teuchos::RCP<Plato::CrsMatrixType> mJacobianU;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianV;

        Teuchos::RCP<Plato::ComputedFields<ElementType::mNumSpatialDims>> mComputedFields;

        Plato::OrdinalVector mStateBcDofs;
        Plato::ScalarVector mStateBcValues;

        rcp<Plato::AbstractSolver> mSolver;

        std::string mPDE; /*!< partial differential equation type */
        std::string mPhysics; /*!< physics used for the simulation */

      public:
        /******************************************************************************/
        Problem(
          Plato::Mesh              aMesh,
          Teuchos::ParameterList & aProblemParams,
          Comm::Machine            aMachine
        );

        /******************************************************************************//**
         * \brief Is criterion independent of the solution state?
         * \param [in] aName Name of criterion.
        **********************************************************************************/
        bool
        criterionIsLinear(
            const std::string & aName
        ) override;

        /******************************************************************************/
        void applyConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector
        );

        /******************************************************************************/
        void applyStateConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector,
          Plato::Scalar aScale
        );

        /******************************************************************************/ /**
        * \brief Output solution to visualization file.
        * \param [in] aFilepath output/visualizaton file path
        **********************************************************************************/
        void output(const std::string &aFilepath) override;

        /******************************************************************************//**
         * \brief Update physics-based parameters within optimization iterations
         * \param [in] aState 2D container of state variables
         * \param [in] aControl 1D container of control variables
        **********************************************************************************/
        void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution);

        /******************************************************************************/
        Plato::Solutions
        solution(
          const Plato::ScalarVector & aControl
        );

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
                  Criterion             aCriterion
        );

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
         * \param [in] aGlobalState 2D view of state variables
         * \param [in] aCriterion criterion to be evaluated
         * \return 1D view - criterion gradient wrt configuration variables
        **********************************************************************************/
        Plato::ScalarVector
        criterionGradientX(
            const Plato::ScalarVector & aControl,
            const Plato::Solutions    & aSolution,
                  Criterion             aCriterion
        );

        private:
        /******************************************************************************//**
         * \brief Return solution database.
         * \return solution database
        **********************************************************************************/
        Plato::Solutions getSolution() const;
    };

} // namespace Parabolic

} // namespace Plato
