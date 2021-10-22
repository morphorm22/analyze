#include "MaterialModel.hpp"
#include "ParseTools.hpp"

namespace Plato {

  template<>
  IsotropicStiffnessFunctor<1>::IsotropicStiffnessFunctor(const Teuchos::ParameterList& aParams)
  {
      typedef Plato::Scalar T;
      auto tYoungsModParams = aParams.sublist("Youngs Modulus");
      T v = Plato::ParseTools::getParam<T>(aParams, "Poissons Ratio"); /*throw if not found*/

      T E0 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c0", 0.0);
      T E1 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c1", 0.0);
      T E2 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c2", 0.0);

      c0[0][0] = E0*((1.0+v)*(1.0-2.0*v))*(1.0-v);
      c1[0][0] = E1*((1.0+v)*(1.0-2.0*v))*(1.0-v);
      c2[0][0] = E2*((1.0+v)*(1.0-2.0*v))*(1.0-v);
  }

  template<>
  IsotropicStiffnessFunctor<2>::IsotropicStiffnessFunctor(const Teuchos::ParameterList& aParams)
  {
      typedef Plato::Scalar T;
      auto tYoungsModParams = aParams.sublist("Youngs Modulus");
      T v = Plato::ParseTools::getParam<T>(aParams, "Poissons Ratio"); /*throw if not found*/

      T E0 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c0"); /*throw if not found*/
      T E1 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c1", 0.0);
      T E2 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c2", 0.0);

      T c = 1.0/((1.0+v)*(1.0-2.0*v));

      T c000 = E0*c*(1.0-v), c001 = E0*c*v, c022 = 1.0/2.0*E0*c*(1.0-2.0*v);
      T c100 = E1*c*(1.0-v), c101 = E1*c*v, c122 = 1.0/2.0*E1*c*(1.0-2.0*v);
      T c200 = E2*c*(1.0-v), c201 = E2*c*v, c222 = 1.0/2.0*E2*c*(1.0-2.0*v);

      c0[0][0] = c000; c0[0][1] = c001;
      c0[1][0] = c001; c0[1][1] = c000;
      c0[2][2] = c022;

      c1[0][0] = c100; c1[0][1] = c101;
      c1[1][0] = c101; c1[1][1] = c100;
      c1[2][2] = c122;

      c2[0][0] = c200; c2[0][1] = c201;
      c2[1][0] = c201; c2[1][1] = c200;
      c2[2][2] = c222;
  }

  template<>
  IsotropicStiffnessFunctor<3>::IsotropicStiffnessFunctor(const Teuchos::ParameterList& aParams)
  {
      typedef Plato::Scalar T;
      auto tYoungsModParams = aParams.sublist("Youngs Modulus");
      T v = Plato::ParseTools::getParam<T>(aParams, "Poissons Ratio"); /*throw if not found*/

      T E0 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c0"); /*throw if not found*/
      T E1 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c1", 0.0);
      T E2 = Plato::ParseTools::getParam<T>(tYoungsModParams, "c2", 0.0);

      T c = 1.0/((1.0+v)*(1.0-2.0*v));

      T c000 = E0*c*(1.0-v), c001 = E0*c*v, c033 = 1.0/2.0*E0*c*(1.0-2.0*v);
      T c100 = E1*c*(1.0-v), c101 = E1*c*v, c133 = 1.0/2.0*E1*c*(1.0-2.0*v);
      T c200 = E2*c*(1.0-v), c201 = E2*c*v, c233 = 1.0/2.0*E2*c*(1.0-2.0*v);

      c0[0][0] = c000; c0[0][1] = c001; c0[0][2] = c001;
      c0[1][0] = c001; c0[1][1] = c000; c0[1][2] = c001;
      c0[2][0] = c001; c0[2][1] = c001; c0[2][2] = c000;
      c0[3][3] = c033; c0[4][4] = c033; c0[5][5] = c033;

      c1[0][0] = c100; c1[0][1] = c101; c1[0][2] = c101;
      c1[1][0] = c101; c1[1][1] = c100; c1[1][2] = c101;
      c1[2][0] = c101; c1[2][1] = c101; c1[2][2] = c100;
      c1[3][3] = c133; c1[4][4] = c133; c1[5][5] = c133;

      c2[0][0] = c200; c2[0][1] = c201; c2[0][2] = c201;
      c2[1][0] = c201; c2[1][1] = c200; c2[1][2] = c201;
      c2[2][0] = c201; c2[2][1] = c201; c2[2][2] = c200;
      c2[3][3] = c233; c2[4][4] = c233; c2[5][5] = c233;
  }
#if 0
  template<typename T>
  IsotropicStiffnessConstant<1,T>::IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams)
  {
      typedef Plato::Scalar RealT;
      T E = Plato::ParseTools::getParam<RealT>(aParams, "Youngs Modulus"); /*throw if not found*/
      T v = Plato::ParseTools::getParam<RealT>(aParams, "Poissons Ratio"); /*throw if not found*/
      T c = 1.0/((1.0+v)*(1.0-2.0*v));

      this->c0[0][0] = E*c*(1.0-v);
  }

  template<typename T>
  IsotropicStiffnessConstant<2,T>::IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams)
  {
      typedef Plato::Scalar RealT;
      T E = Plato::ParseTools::getParam<RealT>(aParams, "Youngs Modulus"); /*throw if not found*/
      T v = Plato::ParseTools::getParam<RealT>(aParams, "Poissons Ratio"); /*throw if not found*/

      T c = 1.0/((1.0+v)*(1.0-2.0*v));

      T c00 = E*c*(1.0-v), c01 = E*c*v, c22 = 1.0/2.0*E*c*(1.0-2.0*v);

      this->c0[0][0] = c00; this->c0[0][1] = c01;
      this->c0[1][0] = c01; this->c0[1][1] = c00;
      this->c0[2][2] = c22;
  }
#endif
#if 0
  template<typename T>
  IsotropicStiffnessConstant<3,T>::IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams)
  {
      typedef Plato::Scalar RealT;
      T E = Plato::ParseTools::getParam<RealT>(aParams, "Youngs Modulus"); /*throw if not found*/
      T v = Plato::ParseTools::getParam<RealT>(aParams, "Poissons Ratio"); /*throw if not found*/

      T c = 1.0/((1.0+v)*(1.0-2.0*v));

      T c00 = E*c*(1.0-v), c01 = E*c*v, c33 = 1.0/2.0*E*c*(1.0-2.0*v);

      this->c0[0][0] = c00; this->c0[0][1] = c01; this->c0[0][2] = c01;
      this->c0[1][0] = c01; this->c0[1][1] = c00; this->c0[1][2] = c01;
      this->c0[2][0] = c01; this->c0[2][1] = c01; this->c0[2][2] = c00;
      this->c0[3][3] = c33; this->c0[4][4] = c33; this->c0[5][5] = c33;
  }
  #endif
#if 0
  template<typename T>
  IsotropicStiffnessConstant<1, T>::IsotropicStiffnessConstant(const T& aYoungsModulus, 
                                                            const T& aPoissonsRatio)
  {
      T c = 1.0/((1.0+aPoissonsRatio)*(1.0-2.0*aPoissonsRatio));
      this->c0[0][0] = aYoungsModulus*c*(1.0-aPoissonsRatio);
  }

  template<typename T>
  IsotropicStiffnessConstant<2, T>::IsotropicStiffnessConstant(const T& aYoungsModulus, 
                                                            const T& aPoissonsRatio)
  {
      T c = 1.0/((1.0+aPoissonsRatio)*(1.0-2.0*aPoissonsRatio));

      T c00 = aYoungsModulus*c*(1.0-aPoissonsRatio);
      T c01 = aYoungsModulus*c*aPoissonsRatio;
      T c22 = 1.0/2.0*aYoungsModulus*c*(1.0-2.0*aPoissonsRatio);

      this->c0[0][0] = c00; this->c0[0][1] = c01;
      this->c0[1][0] = c01; this->c0[1][1] = c00;
      this->c0[2][2] = c22;
  }
#endif
#if 0
  template<typename T>
  IsotropicStiffnessConstant<3, T>::IsotropicStiffnessConstant(const T& aYoungsModulus, 
                                                            const T& aPoissonsRatio)
  {
        T c = 1.0/((1.0+aPoissonsRatio)*(1.0-2.0*aPoissonsRatio));

        T c00 = aYoungsModulus*c*(1.0-aPoissonsRatio);
        T c01 = aYoungsModulus*c*aPoissonsRatio;
        T c33 = 1.0/2.0*aYoungsModulus*c*(1.0-2.0*aPoissonsRatio);

        this->c0[0][0] = c00; this->c0[0][1] = c01; this->c0[0][2] = c01;
        this->c0[1][0] = c01; this->c0[1][1] = c00; this->c0[1][2] = c01;
        this->c0[2][0] = c01; this->c0[2][1] = c01; this->c0[2][2] = c00;
        this->c0[3][3] = c33; this->c0[4][4] = c33; this->c0[5][5] = c33;
  }
  #endif
} // namespace Plato
