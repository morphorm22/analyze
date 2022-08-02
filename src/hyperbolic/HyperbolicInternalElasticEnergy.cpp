#include "hyperbolic/HyperbolicInternalElasticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "hyperbolic/HyperbolicInternalElasticEnergy_def.hpp"

#include "MechanicsElement.hpp"
#include "hyperbolic/ExpInstMacros.hpp"

PLATO_HYPERBOLIC_EXP_INST(Plato::Hyperbolic::InternalElasticEnergy, Plato::MechanicsElement)

#endif