#pragma once

#include "PlatoStaticsTypes.hpp"
#include "Solutions.hpp"
#include "EssentialBCs.hpp"
#include "SpatialModel.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "ComputedField.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/VectorFunction.hpp"
#include "hyperbolic/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Hyperbolic
{
template<typename PhysicsType>
class Problem: public Plato::AbstractProblem
{
  private:

    using Criterion = std::shared_ptr<Plato::Hyperbolic::ScalarFunctionBase>;
    using Criteria  = std::map<std::string, Criterion>;

    using ElementType = typename PhysicsType::ElementType;
    using TopoElementType = typename ElementType::TopoElementType;

    using VectorFunctionType = Plato::Hyperbolic::VectorFunction<PhysicsType>;

    Plato::SpatialModel mSpatialModel;

    VectorFunctionType mPDEConstraint;

    std::shared_ptr<Plato::NewmarkIntegrator> mIntegrator;

    Plato::OrdinalType mNumSteps;
    Plato::Scalar      mTimeStep;

    bool mSaveState;

    Criteria mCriteria;

    Plato::ScalarMultiVector mAdjoints_U;
    Plato::ScalarMultiVector mAdjoints_V;
    Plato::ScalarMultiVector mAdjoints_A;

    Plato::ScalarMultiVector mDisplacement;
    Plato::ScalarMultiVector mVelocity;
    Plato::ScalarMultiVector mAcceleration;

    Plato::ScalarVector mInitDisplacement;
    Plato::ScalarVector mInitVelocity;
    Plato::ScalarVector mInitAcceleration;

    Teuchos::RCP<Plato::CrsMatrixType> mJacobianU;
    Teuchos::RCP<Plato::CrsMatrixType> mJacobianV;
    Teuchos::RCP<Plato::CrsMatrixType> mJacobianA;

    Teuchos::RCP<Plato::ComputedFields<ElementType::mNumSpatialDims>> mComputedFields;

    Plato::EssentialBCs<ElementType> mStateBoundaryConditions;
    Plato::OrdinalVector mStateBcDofs;
    Plato::ScalarVector mStateBcValues;

    Plato::EssentialBCs<ElementType> mStateDotBoundaryConditions;
    Plato::OrdinalVector mStateDotBcDofs;
    Plato::ScalarVector mStateDotBcValues;

    Plato::EssentialBCs<ElementType> mStateDotDotBoundaryConditions;
    Plato::OrdinalVector mStateDotDotBcDofs;
    Plato::ScalarVector mStateDotDotBcValues;

    rcp<Plato::AbstractSolver> mSolver;

    std::string mPDE; /*!< partial differential equation type */
    std::string mPhysics; /*!< physics used for the simulation */
    bool mUForm; /*!< true: displacement-based formulation, false: acceleration-based formulation */

  public:
    Problem(
      Plato::Mesh              aMesh,
      Teuchos::ParameterList & aProblemParams,
      Comm::Machine            aMachine
    ); 

    void parseIntegrator(Teuchos::ParameterList & aProblemParams);

    void allocateStateData();

    void parseCriteria(Teuchos::ParameterList & aProblemParams);

    void parseComputedFields(
      Teuchos::ParameterList & aProblemParams,
      Plato::Mesh              aMesh
    );

    void parseInitialState(Teuchos::ParameterList & aProblemParams);

    void parseLinearSolver(
      Teuchos::ParameterList & aProblemParams,
      Plato::Mesh              aMesh,
      Comm::Machine            aMachine
    );

    void output(const std::string& aFilepath);

    void applyConstraints(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector & aVector
    );

    void applyConstraintType(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector                & aVector,
      const Plato::OrdinalVector               & aBcDofs,
      const Plato::ScalarVector                & aBcValues
    );

    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution);

    Plato::Solutions solution(const Plato::ScalarVector & aControl);

    void forwardStepUForm(
        const Plato::ScalarVector & aControl,
              Plato::Scalar       & aCurrentTime,
              Plato::OrdinalType    aStepIndex
    );

    void forwardStepAForm(
        const Plato::ScalarVector & aControl,
              Plato::Scalar       & aCurrentTime,
              Plato::OrdinalType    aStepIndex
    );

    void computeInitialState(
        const Plato::ScalarVector & aControl
    );

    void constrainFieldsAtBoundary(
              Plato::ScalarVector & aDisplacement,
              Plato::ScalarVector & aVelocity,
              Plato::ScalarVector & aAcceleration,
        const Plato::Scalar         aTime);

    void constrainUFormFieldsAtBoundary(
              Plato::ScalarVector & aVelocity,
              Plato::ScalarVector & aAcceleration,
        const Plato::Scalar         aTime);

    void constrainAFormFieldsAtBoundary(
              Plato::ScalarVector & aDisplacement,
              Plato::ScalarVector & aVelocity,
        const Plato::Scalar         aTime);
    
    Plato::Scalar criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override;

    Plato::Scalar criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override;

    Plato::ScalarVector criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override;

    Plato::ScalarVector criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override;

    Plato::ScalarVector criterionGradient(
      const Plato::ScalarVector & aControl,
      const Plato::Solutions    & aSolution,
            Criterion             aCriterion
    );

    Plato::ScalarVector criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override;

    Plato::ScalarVector criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override;

    Plato::ScalarVector criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion
    );

  private:
    Plato::Solutions getSolution() const override;
};
}

}
