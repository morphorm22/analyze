#ifdef PLATO_CUSTOM_MATERIALS
  #include "J2PlasticityLocalResidualExpFAD.hpp"
#else
  #include "J2PlasticityLocalResidual.hpp"
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 3)
#endif
