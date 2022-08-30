/*
 * FluidsCriterionFactory.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "hyperbolic/fluids/FluidsCriterionFactory.hpp"
#include "hyperbolic/fluids/FluidsCriterionFactory_def.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<3>>;
#endif
