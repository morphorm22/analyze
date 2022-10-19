#pragma once

#include <memory>

#include "MultipointConstraints.hpp"

namespace Plato {

template <typename ClassT>
using rcp = std::shared_ptr<ClassT>;

enum class LinearSystemType {
    SYMMETRIC_POSITIVE_DEFINITE,
    SYMMETRIC_INDEFINITE,
    SYMMETRIC_PATTERN
};

/******************************************************************************//**
 * \brief Abstract solver interface

  Note that the solve() function takes 'native' matrix and vector types.  A next
  step would be to adopt generic matrix and vector interfaces that we can wrap
  around Epetra types, Tpetra types, Kokkos view-based types, etc.
**********************************************************************************/
class AbstractSolver
{
  protected:
    std::shared_ptr<Plato::MultipointConstraints>   mSystemMPCs;

    Plato::Scalar mAlpha;

    AbstractSolver();
    AbstractSolver(const Teuchos::ParameterList & aSolverParams);

    void parse(const Teuchos::ParameterList & aSolverParams);

    virtual void innerSolve(
        Plato::CrsMatrix<Plato::OrdinalType> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) = 0;

    virtual ~AbstractSolver() = default;

  public:
    AbstractSolver(const Teuchos::ParameterList & aSolverParams, std::shared_ptr<Plato::MultipointConstraints> aMPCs);

    void solve(
        Plato::CrsMatrix<int> aAf,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB,
        bool                  aAdjointFlag = false);
};
} // end namespace Plato
