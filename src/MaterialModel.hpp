#pragma once

#include <Teuchos_ParameterList.hpp>
#include "PlatoStaticsTypes.hpp"
#include "ParseTools.hpp"

namespace Plato {

  enum class MaterialModelType { Linear, Nonlinear, Expression };

  /******************************************************************************/
  /*!
    \brief class for mappings from scalar to scalar
  */
  class ScalarFunctor
  /******************************************************************************/
  {
    Plato::Scalar c0, c1, c2;

    public:
      /******************************************************************************//**
       * \brief Constructor for zero functor
       *   Functor returns 0.0
       * unit test: PlatoMaterialModel_ScalarFunctor
      **********************************************************************************/
      ScalarFunctor() : c0(0.0), c1(0.0), c2(0.0) {}

      /******************************************************************************//**
       * \brief Constructor for constant functor
       *   Functor returns \a aVal
       * \param [in] aVal Constant value
       * unit test: PlatoMaterialModel_ScalarFunctor
      **********************************************************************************/
      ScalarFunctor(Plato::Scalar aVal) : c0(aVal), c1(0.0), c2(0.0) {}


      /******************************************************************************//**
       * \brief Constructor for quadratic functor
       *   Functor returns \f$ y(x) = c0 + c1 x + c2 x^2 \f$
       * \param [in] ParameterList with "c0", "c1", and "c2" Parameters
       * unit test: PlatoMaterialModel_ScalarFunctor
      **********************************************************************************/
      ScalarFunctor(Teuchos::ParameterList& aParams) : c0(0.0), c1(0.0), c2(0.0)
      {
          if (aParams.isType<Plato::Scalar>("c0"))
          {
              c0 = aParams.get<Plato::Scalar>("c0");
          }
          else
          {
              THROWERR("Missing required parameter 'c0'");
          }

          if (aParams.isType<Plato::Scalar>("c1"))
          {
              c1 = aParams.get<Plato::Scalar>("c1");
          }

          if (aParams.isType<Plato::Scalar>("c2"))
          {
              c2 = aParams.get<Plato::Scalar>("c2");
          }
      }
      /******************************************************************************//**
       * \brief Quadratic functor
       * \return \f$ y(x) = c0 + c1 x + c2 x^2 \f$
       * \param [in] aInput  \f$ x \f$ value in expression.
       * unit test: PlatoMaterialModel_ScalarFunctor
      **********************************************************************************/
      template<typename TScalarType>
      DEVICE_TYPE inline TScalarType
      operator()( TScalarType aInput ) const {
          TScalarType tRetVal(aInput);
          tRetVal *= c1;
          tRetVal += c0;
          tRetVal += aInput*aInput*c2;
          return tRetVal;
      }
  };

  /******************************************************************************/
  /*!
    \brief class for mappings from scalar to 2nd rank tensor
  */
  template<int SpatialDim>
  class TensorFunctor
  /******************************************************************************/
  {
    Plato::Scalar c0[SpatialDim][SpatialDim];
    Plato::Scalar c1[SpatialDim][SpatialDim];
    Plato::Scalar c2[SpatialDim][SpatialDim];

    public:

      /******************************************************************************//**
       * \brief Constructor for zero 2nd rank tensor functor
       *   Functor returns 0.0 for given tensor indices
       * unit test: PlatoMaterialModel_TensorFunctor
      **********************************************************************************/
      TensorFunctor() : c0{{0.0}}, c1{{0.0}}, c2{{0.0}} {}

      /******************************************************************************//**
       * \brief Constructor for constant diagonal 2nd rank tensor functor
       *   Functor returns \a aValue for given tensor indices if \F$i==j\f$, 0.0 otherwise.
       * unit test: PlatoMaterialModel_TensorFunctor
      **********************************************************************************/
      TensorFunctor(Plato::Scalar aValue) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          for (int iDim=0; iDim<SpatialDim; iDim++)
          {
              c0[iDim][iDim] = aValue;
          }
      }

      /******************************************************************************//**
       * \brief Constructor for up to quadratic 2nd rank tensor functor
       *   Functor returns \f$ y(x, i, j) = c0(i,j) + c1(i,j) x + c2(i,j) x^2 \f$
       * \param [in] ParameterList with "c0ij", "c1ij", and "c2ij" Parameters
       * unit test: PlatoMaterialModel_TensorFunctor
      **********************************************************************************/
      TensorFunctor(Teuchos::ParameterList& aParams);

      /******************************************************************************//**
       * \brief Functor for up to quadratic 2nd rank tensor constant
       * \return \f$ y(x, i, j) = c0(i,j) + c1(i,j) x + c2(i,j) x^2 \f$
       * \param [in] aInput  \f$ x \f$ value in expression.
       * \param [in] i \f$ i \f$ value in expression.
       * \param [in] j \f$ j \f$ value in expression.
       * unit test: PlatoMaterialModel_TensorFunctor
      **********************************************************************************/
      template<typename TScalarType>
      DEVICE_TYPE inline TScalarType
      operator()( TScalarType aInput, Plato::OrdinalType i, Plato::OrdinalType j ) const {
          TScalarType tRetVal(aInput);
          tRetVal *= c1[i][j];
          tRetVal += c0[i][j];
          tRetVal += aInput*aInput*c2[i][j];
          return tRetVal;
      }
  };

  /******************************************************************************/
  /*!
    \brief class for 2nd rank tensor constant
  */
  template<int SpatialDim>
  class TensorConstant
  /******************************************************************************/
  {
    Plato::Scalar c0[SpatialDim][SpatialDim];

    public:

      /******************************************************************************//**
       * \brief Constructor for zero 2nd rank tensor constant
       *   Functor returns 0.0 for given tensor indices
       * unit test: PlatoMaterialModel_TensorConstant
      **********************************************************************************/
      TensorConstant() : c0{{0.0}} {}

      /******************************************************************************//**
       * \brief Constructor for diagonal 2nd rank tensor constant
       *   Functor returns \a aValue for given tensor indices if \F$i==j\f$, 0.0 otherwise.
       * unit test: PlatoMaterialModel_TensorConstant
      **********************************************************************************/
      TensorConstant(Plato::Scalar aValue) : c0{{0.0}}
      {
          for (int iDim=0; iDim<SpatialDim; iDim++)
          {
              c0[iDim][iDim] = aValue;
          }
      }

      /******************************************************************************//**
       * \brief Constructor for 2nd rank tensor constant
       *   Functor returns \f$ y(i, j) = c0(i,j) \f$
       * \param [in] ParameterList with "c0ij" Parameters
       * unit test: PlatoMaterialModel_TensorConstant
      **********************************************************************************/
      TensorConstant(Teuchos::ParameterList& aParams);

      /******************************************************************************//**
       * \brief Functor for 2nd rank tensor constant
       * \return \f$ y(i, j) = c0(i,j) \f$
       * \param [in] i \f$ i \f$ value in expression.
       * \param [in] j \f$ j \f$ value in expression.
       * unit test: PlatoMaterialModel_TensorFunctor
      **********************************************************************************/
      DEVICE_TYPE inline Plato::Scalar
      operator()(Plato::OrdinalType i, Plato::OrdinalType j ) const {
          return c0[i][j];
      }
  };

  /******************************************************************************/
  /*!
    \brief class for mappings from scalar to 4th rank voigt tensor
  */
  template<int SpatialDim>
  class Rank4VoigtFunctor
  /******************************************************************************/
  {
    protected:
      static constexpr Plato::OrdinalType NumVoigtTerms = (SpatialDim == 3) ? 6 :
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));

      Plato::Scalar c0[NumVoigtTerms][NumVoigtTerms];
      Plato::Scalar c1[NumVoigtTerms][NumVoigtTerms];
      Plato::Scalar c2[NumVoigtTerms][NumVoigtTerms];

    public:

      /******************************************************************************//**
       * \brief Constructor for zero 4th rank voigt tensor functor
       *   Functor returns 0.0 for given tensor indices
       * unit test: PlatoMaterialModel_Rank4VoigtFunctor
      **********************************************************************************/
      Rank4VoigtFunctor() : c0{{0.0}}, c1{{0.0}}, c2{{0.0}} {}

      /******************************************************************************//**
       * \brief Constructor for zero 4th rank voigt tensor functor
       *   Functor returns \f$ y(x, i, j) = c0(i,j) + c1(i,j) x + c2(i,j) x^2 \f$
       * \param [in] ParameterList with "c0ij", "c1ij", and "c2ij" Parameters
       * unit test: PlatoMaterialModel_Rank4VoigtConstantFunctor
      **********************************************************************************/
      Rank4VoigtFunctor(Teuchos::ParameterList& aParams);

      /******************************************************************************//**
       * \brief Functor for quadratic 4th rank tensor
       * \return \f$ y(x, i, j) = c0(i,j) + c1(i,j) x + c2(i,j) x^2 \f$
       * \param [in] aInput  \f$ x \f$ value in expression.
       * \param [in] i \f$ i \f$ value in expression.
       * \param [in] j \f$ j \f$ value in expression.
       * unit test: PlatoMaterialModel_Rank4VoigtConstantFunctor
      **********************************************************************************/
      template<typename TScalarType>
      DEVICE_TYPE inline TScalarType
      operator()( TScalarType aInput, Plato::OrdinalType i, Plato::OrdinalType j ) const {
          TScalarType tRetVal(aInput);
          tRetVal *= c1[i][j];
          tRetVal += c0[i][j];
          tRetVal += aInput*aInput*c2[i][j];
          return tRetVal;
      }
  };

  /******************************************************************************/
  /*!
    \brief class for tensor constant
  */
  template<int SpatialDim, typename T = Plato::Scalar>
  class Rank4VoigtConstant
  {
    protected:

        static constexpr Plato::OrdinalType NumVoigtTerms = (SpatialDim == 3) ? 6 :
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));

        T c0[NumVoigtTerms][NumVoigtTerms];
    
    public:

      /******************************************************************************//**
       * \brief Constructor for zero 4th rank voigt tensor constant
       *   Functor returns 0.0 for given tensor indices
       * unit test: PlatoMaterialModel_Rank4VoigtConstant
      **********************************************************************************/
      Rank4VoigtConstant() : c0{{0.0}} {}

      /******************************************************************************//**
       * \brief Constructor for 4th rank voigt tensor constant
       *   Functor returns \f$ y(i, j) = c0(i,j) \f$
       * \param [in] ParameterList with "c0ij" Parameters
       * unit test: PlatoMaterialModel_Rank4VoigtConstant
      **********************************************************************************/
      Rank4VoigtConstant(Teuchos::ParameterList& aParams);

      /******************************************************************************//**
       * \brief Functor for 4th rank tensor constant
       * \return \f$ y(i, j) = c0(i,j) \f$
       * \param [in] i \f$ i \f$ value in expression.
       * \param [in] j \f$ j \f$ value in expression.
       * unit test: PlatoMaterialModel_Rank4VoigtConstant
      **********************************************************************************/
      DEVICE_TYPE inline T
      operator()(Plato::OrdinalType i, Plato::OrdinalType j ) const {
       return c0[i][j];
      }
  };
#if 0
  template<typename T>
  class Rank4VoigtConstant<1,T>
  /******************************************************************************/
  {
    protected:
      static constexpr Plato::OrdinalType NumVoigtTerms = 1;

      T c0[NumVoigtTerms][NumVoigtTerms];

    public:

      Rank4VoigtConstant() : c0{{0.0}} {}

      Rank4VoigtConstant(Teuchos::ParameterList& aParams);

      DEVICE_TYPE inline T
      operator()(Plato::OrdinalType i, Plato::OrdinalType j ) const {
          return c0[i][j];
      }
  };
#endif

  template<typename T>
  class Rank4VoigtConstant<2,T>
  /******************************************************************************/
  {
    protected:
      static constexpr Plato::OrdinalType NumVoigtTerms = 3;

      T c0[NumVoigtTerms][NumVoigtTerms];

    public:

      Rank4VoigtConstant() : c0{{0.0}} {}

      Rank4VoigtConstant(Teuchos::ParameterList& aParams)
      {
          typedef Plato::Scalar RealT;
          c0[0][0] = Plato::ParseTools::getParam<RealT>(aParams, "c11" /*throw if not found*/);
          c0[1][1] = Plato::ParseTools::getParam<RealT>(aParams, "c22", /*default=*/ c0[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<RealT>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c0[2][2] = Plato::ParseTools::getParam<RealT>(aParams, "c33" /*throw if not found*/);
      }

      DEVICE_TYPE inline T
      operator()(Plato::OrdinalType i, Plato::OrdinalType j ) const {
          return c0[i][j];
      }
  };
  template<typename T>
  class Rank4VoigtConstant<3,T>
  /******************************************************************************/
  {
    protected:
      static constexpr Plato::OrdinalType NumVoigtTerms = 6;

      T c0[NumVoigtTerms][NumVoigtTerms];

    public:

      Rank4VoigtConstant() : c0{{0.0}} {}

      Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
      {
          typedef Plato::Scalar RealT;
          c0[0][0] = Plato::ParseTools::getParam<RealT>(aParams, "c11" /*throw if not found*/);
          c0[1][1] = Plato::ParseTools::getParam<RealT>(aParams, "c22", /*default=*/ c0[0][0]);
          c0[2][2] = Plato::ParseTools::getParam<RealT>(aParams, "c33", /*default=*/ c0[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<RealT>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c0[0][2] = Plato::ParseTools::getParam<RealT>(aParams, "c13", /*default=*/ c0[0][1]); c0[2][0] = c0[0][2];
          c0[1][2] = Plato::ParseTools::getParam<RealT>(aParams, "c23", /*default=*/ c0[0][1]); c0[2][1] = c0[1][2];
          c0[3][3] = Plato::ParseTools::getParam<RealT>(aParams, "c44" /*throw if not found*/);
          c0[4][4] = Plato::ParseTools::getParam<RealT>(aParams, "c55", c0[3][3]);
          c0[5][5] = Plato::ParseTools::getParam<RealT>(aParams, "c66", c0[3][3]);
      }

      DEVICE_TYPE inline T
      operator()(Plato::OrdinalType i, Plato::OrdinalType j ) const {
          return c0[i][j];
      }
  };

  /******************************************************************************/
  /*!
    \brief class for mappings from scalar to 4th rank voigt tensor
  */
  template<int SpatialDim>
  class IsotropicStiffnessFunctor : public Rank4VoigtFunctor<SpatialDim>
  /******************************************************************************/
  {
    public:
      /******************************************************************************//**
       * \brief Constructor for isotropic elastic 4th rank voigt tensor functor
       *   Functor returns \f$ y(x, i, j) = c0(i,j) + c1(i,j) x + c2(i,j) x^2 \f$
       * where c0, c1, and c2 coefficients are determined from the user provided constant
       * Poisson's ratio and the temperature dependent Young's modulus.
       * \param [in] ParameterList with "Poissons Ratio" Parameter and "Youngs Modulus" ParameterList
       * unit test: PlatoMaterialModel_IsotropicStiffnessFunctor
      **********************************************************************************/
      IsotropicStiffnessFunctor(const Teuchos::ParameterList& aParams);
  };


  /******************************************************************************/
  /*!
    \brief class for tensor constant
  */
  template<int SpatialDim, typename T = Plato::Scalar>
  class IsotropicStiffnessConstant : public Rank4VoigtConstant<SpatialDim, T>
  /******************************************************************************/
  {
    public:
      /******************************************************************************//**
       * \brief Constructor for isotropic elastic 4th rank voigt tensor constant
       *   Functor returns \f$ y(i, j) = c0(i,j) \f$
       * \param [in] ParameterList with "c0ij" Parameters
       * unit test: PlatoMaterialModel_IsotropicStiffnessConstant
       **********************************************************************************/
       IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams){}
       IsotropicStiffnessConstant(const T& aYoungsModulus, const T& aPoissonsRatio){}
  };
#if 0  
  template<typename T>
  class IsotropicStiffnessConstant<1,T> : public Rank4VoigtConstant<1,T>
  /******************************************************************************/
  {
    public:
      IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams);
      IsotropicStiffnessConstant(const T& aYoungsModulus, const T& aPoissonsRatio);
  };

  template<typename T>
  class IsotropicStiffnessConstant<2,T> : public Rank4VoigtConstant<2,T>
  /******************************************************************************/
  {
    public:
      IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams);
      IsotropicStiffnessConstant(const T& aYoungsModulus, const T& aPoissonsRatio);
  };
#endif
  template<typename T>
  class IsotropicStiffnessConstant<3,T> : public Rank4VoigtConstant<3,T>
  /******************************************************************************/
  {
    public:
      IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams)
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

      IsotropicStiffnessConstant(const T& aYoungsModulus, const T& aPoissonsRatio)
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

  };

  /******************************************************************************/
  /*!
    \brief Base class for material models
  */
    template<int SpatialDim>
    class MaterialModel
  /******************************************************************************/
  {
      std::map<std::string, Plato::Scalar>                         mScalarConstantsMap;
      std::map<std::string, Plato::TensorConstant<SpatialDim>>     mTensorConstantsMap;
      std::map<std::string, Plato::Rank4VoigtConstant<SpatialDim>> mRank4VoigtConstantsMap;

      std::map<std::string, Plato::ScalarFunctor>                 mScalarFunctorsMap;
      std::map<std::string, Plato::TensorFunctor<SpatialDim>>     mTensorFunctorsMap;
      std::map<std::string, Plato::Rank4VoigtFunctor<SpatialDim>> mRank4VoigtFunctorsMap;

      Plato::MaterialModelType mType;
      std::string mExpression;

    public:

      /******************************************************************************//**
       * \brief Default constructor for Plato::MaterialModel.
      **********************************************************************************/
      MaterialModel() : mType(Plato::MaterialModelType::Linear) {}

      /******************************************************************************//**
       * \brief Constructor for Plato::MaterialModel base class
       * \param [in] ParameterList with optional "Temperature Dependent" bool Parameter
       * unit test: PlatoMaterialModel_MaterialModel
      **********************************************************************************/
      MaterialModel(const Teuchos::ParameterList& paramList) 
      {
          this->mType = Plato::MaterialModelType::Linear;
          if (paramList.isType<bool>("Temperature Dependent"))
          {
              if (paramList.get<bool>("Temperature Dependent")) {
                  this->mType = Plato::MaterialModelType::Nonlinear;
              }
          }
          if (paramList.isSublist("Custom Elastic Stiffness")) 
          {
              this->mType = Plato::MaterialModelType::Expression;
              auto tCustomElasticSubList = paramList.sublist("Custom Elastic Stiffness");
              if(tCustomElasticSubList.isType<double>("E0"))
              {          
                  this->setScalarConstant("E0", tCustomElasticSubList.get<double>("E0"));
              }
              if(tCustomElasticSubList.isType<std::string>("Expression"))
              {          
                  this->expression(tCustomElasticSubList.get<std::string>("Expression"));
              }
              if(tCustomElasticSubList.isType<double>("Poissons Ratio"))
              {          
                  this->setScalarConstant("Poissons Ratio", tCustomElasticSubList.get<double>("Poissons Ratio"));
              }
              if(tCustomElasticSubList.isType<double>("Density"))
              {          
                  this->setScalarConstant("Density", tCustomElasticSubList.get<double>("Density"));
              }
          }
      }

      Plato::MaterialModelType type() const { return this->mType; }
      std::string expression() const { return this->mExpression; }
      void expression(const std::string aString) { this->mExpression = aString; }

      // getters
      //

      // scalar constant
      Plato::Scalar getScalarConstant(std::string aConstantName)
      { return mScalarConstantsMap[aConstantName]; }

      // Tensor constant
      Plato::TensorConstant<SpatialDim> getTensorConstant(std::string aConstantName)
      { return mTensorConstantsMap[aConstantName]; }

      // Rank4Voigt constant
      Plato::Rank4VoigtConstant<SpatialDim> getRank4VoigtConstant(std::string aConstantName)
      { return mRank4VoigtConstantsMap[aConstantName]; }

      // scalar functor
      Plato::ScalarFunctor getScalarFunctor(std::string aFunctorName)
      { return mScalarFunctorsMap[aFunctorName]; }

      // tensor functor
      Plato::TensorFunctor<SpatialDim> getTensorFunctor(std::string aFunctorName)
      { return mTensorFunctorsMap[aFunctorName]; }

      // Rank4Voigt functor
      Plato::Rank4VoigtFunctor<SpatialDim> getRank4VoigtFunctor(std::string aFunctorName)
      { return mRank4VoigtFunctorsMap[aFunctorName]; }


      // setters
      //

      // scalar constant
      void setScalarConstant(std::string aConstantName, Plato::Scalar aConstantValue)
      { mScalarConstantsMap[aConstantName] = aConstantValue; }

      // tensor constant
      void setTensorConstant(std::string aConstantName, Plato::TensorConstant<SpatialDim> aConstantValue)
      { mTensorConstantsMap[aConstantName] = aConstantValue; }

      // Rank4Voigt constant
      void setRank4VoigtConstant(std::string aConstantName, Plato::Rank4VoigtConstant<SpatialDim> aConstantValue)
      { mRank4VoigtConstantsMap[aConstantName] = aConstantValue; }

      // scalar functor
      void setScalarFunctor(std::string aFunctorName, Plato::ScalarFunctor aFunctorValue)
      { mScalarFunctorsMap[aFunctorName] = aFunctorValue; }

      // tensor functor
      void setTensorFunctor(std::string aFunctorName, Plato::TensorFunctor<SpatialDim> aFunctorValue)
      { mTensorFunctorsMap[aFunctorName] = aFunctorValue; }

      // Rank4Voigt functor
      void setRank4VoigtFunctor(std::string aFunctorName, Plato::Rank4VoigtFunctor<SpatialDim> aFunctorValue)
      { mRank4VoigtFunctorsMap[aFunctorName] = aFunctorValue; }


      /******************************************************************************/
      /*!
        \brief create either scalar constant or scalar functor from input
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseScalar(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setScalarFunctor(aName, Plato::ScalarFunctor(tValue));
            }
            else
            {
                this->setScalarConstant(aName, tValue);
            }
          }
          else
          if( aParamList.isSublist(aName) )
          {
            if (this->mType == Plato::MaterialModelType::Linear)
            {
                std::stringstream err;
                err << "Found a temperature dependent constant in a linear model." << std::endl;
                err << "Models must be declared temperature dependent." << std::endl;
                err << "Set Parameter 'temperature dependent' to 'true'." << std::endl;
                THROWERR(err.str());
            }
            auto tList = aParamList.sublist(aName);
            this->setScalarFunctor(aName, Plato::ScalarFunctor(tList));
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as either a Parameter or ParameterList";
              THROWERR(err.str());
          }
      }
      /******************************************************************************/
      /*!
        \brief create scalar constant.  Add default if not found
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void
      parseScalarConstant(
        std::string aName,
        const Teuchos::ParameterList& aParamList,
        Plato::Scalar aDefaultValue)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
              auto tValue= aParamList.get<Plato::Scalar>(aName);
              this->setScalarConstant(aName, tValue);
          }
          else
          {
              this->setScalarConstant(aName, aDefaultValue);
          }
      }

      /******************************************************************************/
      /*!
        \brief create scalar constant.  Throw if not found.
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseScalarConstant(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            this->setScalarConstant(aName, tValue);
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as a Parameter of type 'double'";
              THROWERR(err.str());
          }
      }

      /******************************************************************************/
      /*!
        \brief create either tensor constant or tensor functor from input
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseTensor(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setTensorFunctor(aName, Plato::TensorFunctor<SpatialDim>(tValue));
            }
            else
            {
                this->setTensorConstant(aName, Plato::TensorConstant<SpatialDim>(tValue));
            }
          }
          else
          if( aParamList.isSublist(aName) )
          {
            auto tList = aParamList.sublist(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setTensorFunctor(aName, Plato::TensorFunctor<SpatialDim>(tList));
            }
            else
            {
                this->setTensorConstant(aName, Plato::TensorConstant<SpatialDim>(tList));
            }
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as either a Parameter or ParameterList";
              THROWERR(err.str());
          }
      }

      /******************************************************************************/
      /*!
        \brief create either Rank4Voigt constant or Rank4Voigt functor from input
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseRank4Voigt(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if( aParamList.isSublist(aName) )
          {
              auto tList = aParamList.sublist(aName);
              if (this->mType == Plato::MaterialModelType::Linear)
              {
                  this->setRank4VoigtConstant(aName, Plato::Rank4VoigtConstant<SpatialDim>(tList));
              }
              else
              {
                  this->setRank4VoigtFunctor(aName, Plato::Rank4VoigtFunctor<SpatialDim>(tList));
              }
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as a ParameterList";
              THROWERR(err.str());
          }
      }
  };
} // namespace Plato
