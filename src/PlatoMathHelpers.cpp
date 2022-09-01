#include "PlatoMathHelpers_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "PlatoMathHelpers_def.hpp"

#include "PlatoStaticsTypes.hpp"

#include <Teuchos_RCP.hpp>

template
void Plato::MatrixTimesVectorPlusVector<double>(const Teuchos::RCP<Plato::CrsMatrixType> &,
                                 const Plato::ScalarVectorT<double> &,
                                 const Plato::ScalarVectorT<double> &);

template
void Plato::VectorTimesMatrixPlusVector<double>(const Plato::ScalarVectorT<double> &,
                                         const Teuchos::RCP<Plato::CrsMatrixType> &,
                                         const Plato::ScalarVectorT<double> &);

#endif
