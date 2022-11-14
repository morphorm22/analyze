#include "elliptic/Volume_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/Volume_def.hpp"

#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::ThermomechanicsElement)

#endif

