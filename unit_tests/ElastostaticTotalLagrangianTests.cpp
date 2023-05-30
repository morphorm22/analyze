/*
 * ElastostaticTotalLagrangianTests.cpp
 *
 *  Created on: May 10, 2023
 */

// trilinos includes
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// unit test includes
#include "util/PlatoTestHelpers.hpp"

// plato
#include "Tri3.hpp"
#include "BLAS1.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "WorksetBase.hpp"
#include "MaterialModel.hpp"
#include "GradientMatrix.hpp"
#include "MechanicsElement.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace mechanical
{

/// @enum property
/// @brief supported mechanical material property enums
enum struct property
{
  /// @brief Supported mechanical material property enums
  YOUNGS_MODULUS=0, 
  POISSON_RATIO=1, 
  MASS_DENSITY=2, 
  LAME_LAMBDA=3, 
  LAME_MU=4,
  TO_ERSATZ_MATERIAL_EXPONENT=5,
  TO_MIN_ERSATZ_MATERIAL_VALUE=6
};

/// @struct PropEnum
/// @brief interface between input mechanical material property string and supported mechanical material property enum
struct PropEnum
{
private:
  /// @brief map from input mechanical material property string to supported mechanical material property enum
  std::unordered_map<std::string,Plato::mechanical::property> s2e = {
      {"youngs modulus"  ,Plato::mechanical::property::YOUNGS_MODULUS},
      {"poissons ratio"  ,Plato::mechanical::property::POISSON_RATIO},
      {"density"         ,Plato::mechanical::property::MASS_DENSITY},
      {"lame lambda"     ,Plato::mechanical::property::LAME_LAMBDA},
      {"lame mu"         ,Plato::mechanical::property::LAME_MU},
      {"penalty exponent",Plato::mechanical::property::TO_ERSATZ_MATERIAL_EXPONENT},
      {"minimum value"   ,Plato::mechanical::property::TO_MIN_ERSATZ_MATERIAL_VALUE}    
  };

public:
  /// @fn get
  /// @brief Return mechanical property enum associated with input string, 
  ///   throw if requested mechanical property is not supported
  /// @param [in] aInput property identifier
  /// @return mechanical property enum
  Plato::mechanical::property 
  get(
    const std::string &aInput
  ) const
  {
    auto tLower = Plato::tolower(aInput);
    auto tItr = s2e.find(tLower);
    if( tItr == s2e.end() ){
        auto tMsg = this->getErrorMsg(tLower);
        ANALYZE_THROWERR(tMsg)
    }
    return tItr->second;
  }

private:
  /// @fn getErrorMsg
  /// @brief return error message
  /// @param [in] aInProperty property name enter in input deck
  /// @return error message string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  ) const
  {
    auto tMsg = std::string("Did not find matching enum for input mechanical property '") 
            + aInProperty + "'. Supported mechanical property keywords are: ";
    for(const auto& tPair : s2e)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
  }
};

/// @enum material
/// @brief supported mechanical material enums
enum struct material
{
  /// @brief Supported mechanical material enums
  KIRCHHOFF=0,
  NEO_HOOKEAN=1,
};

/// @struct PropEnum
/// @brief interface between input mechanical material string and supported mechanical material enum
struct MaterialEnum
{
private:
  /// @brief map from input mechanical material string to supported mechanical material enum
  std::unordered_map<std::string,Plato::mechanical::material> s2e = {
    {"kirchhoff"  ,Plato::mechanical::material::KIRCHHOFF},
    {"neo-hookean",Plato::mechanical::material::NEO_HOOKEAN},
  };

public:
  /// @fn get
  /// @brief Return mechanical material enum associated with input string, 
  ///   throw if requested mechanical material is not supported
  /// @param [in] aInput mechanical material identifier
  /// @return mechanical material enum
  Plato::mechanical::material 
  get(
    const std::string &aInput
  ) const
  {
    auto tLower = Plato::tolower(aInput);
    auto tItr = s2e.find(tLower);
    if( tItr == s2e.end() ){
        auto tMsg = this->getErrorMsg(tLower);
        ANALYZE_THROWERR(tMsg)
    }
    return tItr->second;
  }

private:
  /// @fn getErrorMsg
  /// @brief return error message
  /// @param [in] aInProperty property name enter in input deck
  /// @return error message string
  std::string
  getErrorMsg(
    const std::string & aInProperty
  ) const
  {
    auto tMsg = std::string("Did not find matching enum for input mechanical material '") 
            + aInProperty + "'. Supported mechanical material keywords are: ";
    for(const auto& tPair : s2e)
    {
        tMsg = tMsg + "'" + tPair.first + "', ";
    }
    auto tSubMsg = tMsg.substr(0,tMsg.size()-2);
    return tSubMsg;
  }
};

}
// namespace mechanical

/// @class StateGradient
/// @brief Computes state gradient:
/// \f[ 
///     \nabla\mathbf{U}=\frac{\partial\mathbf{U}}{\partial\mathbf{X}}
/// \f]
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StateGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode  = ElementType::mNumDofsPerNode;
  /// @brief number of spatial dimensions 
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;

public:
  /// @fn operator()()
  /// @brief compute state gradient
  /// @param [in]     aCellIndex local element ordinal
  /// @param [in]     aStates    2D state workset
  /// @param [in]     aGradient  gradient functions
  /// @param [in,out] aStateGrad state gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::OrdinalType                                               & aCellIndex,
    const Plato::ScalarMultiVectorT<StateScalarType>                       & aStates,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad
  ) const
  {
    for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; tNode++)
    {
      for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
      {
        Plato::OrdinalType tDof = (tNode * mNumDofsPerNode) + tDimI;
        for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
        {
          aStateGrad(tDimI,tDimJ) += aStates(aCellIndex,tDof) * aGradient(tNode,tDimJ);
        }
      }
    }
  }
};

/// @class DeformationGradient
/// @brief Computes deformation gradient:
/// \f[ 
///   F_{ij}=\frac{\partial{x}_i}{\partial{X}_j}=\frac{\partial{u}_i}{\partial{X}_j} + \delta_{ij}
/// \f]
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class DeformationGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @fn operator()()
  /// @brief compute deformation gradient
  /// @param [in]     aStateGrad state gradient
  /// @param [in,out] aDefGrad   deformation gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aDefGrad(tDimI,tDimJ) += aStateGrad(tDimI,tDimJ);
      }
      aDefGrad(tDimI,tDimI) += 1.0;
    }
  }
};

/// @class RightDeformationTensor
/// @brief Computes right deformation tensor: 
/// \f[ 
///   C_{ij}=F_{ik}^{T}F_{kj}
/// \f]
/// where \f$F_{ij}\f$ is the deformation gradient \n
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class RightDeformationTensor
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

public:
  /// @fn operator()()
  /// @brief 
  /// @param [in]     aDefGradT  deformation gradient transpose
  /// @param [in]     aDefGrad   deformation gradient 
  /// @param [in,out] aDefTensor deformation tensor 
  /// @return 
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGradT,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          aDefTensor(tDimI,tDimJ) += aDefGradT(tDimI,tDimK) * aDefGrad(tDimK,tDimJ);
        }
      }
    }
  }
};

/// @class GreenLagrangeStrainTensor
/// @brief Computes Green-Lagrange strain tensor:
/// \f[ 
///   E_{ij}=\frac{1}{2}\left(F_{ik}^{T}F_{kj}-\delta_{ij}\right)
/// \f]
/// where \f$F_{ij}\f$ is the deformation gradient and \f$\delta_{ij}\f$ is the Kronecker delta
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class GreenLagrangeStrainTensor
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

public:
  /// @fn operator()()
  /// @brief computes the green-lagrange strain tensor
  /// @param [in]     aStateGrad    state gradient
  /// @param [in,out] aStrainTensor green-lagrange strain tesnor
  /// @return 
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStrainTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        aStrainTensor(tDimI,tDimJ) += 0.5 * ( aStateGrad(tDimI,tDimJ) + aStateGrad(tDimJ,tDimI) );
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++)
        {
          aStrainTensor(tDimI,tDimJ) += 0.5 * ( aStateGrad(tDimK,tDimI) * aStateGrad(tDimK,tDimJ) );
        }
      }
    }
  }
};

/// @class MaterialKirchhoff
/// @brief material interface for a kirchhoff material model
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialKirchhoff : public Plato::MaterialModel<EvaluationType>
{
private:
    /// @brief topological element typename
    using ElementType = typename EvaluationType::ElementType;
    /// @brief number of spatial dimensions 
    static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
    /// @brief map from input string to supported mechanical property
    Plato::mechanical::PropEnum mS2E;
    /// @brief map from mechanical property enum to list of property values in string format
    std::unordered_map<Plato::mechanical::property,std::vector<std::string>> mProperties;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName input material parameter list name
  /// @param [in] aParamList    input material parameter list
  MaterialKirchhoff(
      const std::string            & aMaterialName,
      const Teuchos::ParameterList & aParamList
  )
  {
    // set material input parameter list name
    this->name(aMaterialName);
    // parse youngs modulus
    this->parseScalar("Youngs Modulus", aParamList);
    auto tYoungsModulus = this->getScalarConstant("Youngs Modulus");
    mProperties[mS2E.get("Youngs Modulus")].push_back( std::to_string(tYoungsModulus) );
    // parse poissons ratio
    this->parseScalar("Poissons Ratio", aParamList);
    auto tPoissonsRatio = this->getScalarConstant("Poissons Ratio");
    mProperties[mS2E.get("Poissons Ratio")].push_back( std::to_string(tPoissonsRatio) );
    // compute Lame constants
    auto tMu = tYoungsModulus / ( 2.*(1.+tPoissonsRatio) );
    mProperties[mS2E.get("Lame Mu")].push_back( std::to_string(tMu) );
    auto tLambda = (tPoissonsRatio * tYoungsModulus) / ( (1.+tPoissonsRatio)*(1.-2.*tPoissonsRatio) );
    mProperties[mS2E.get("Lame Lambda")].push_back( std::to_string(tLambda) );
  }
  /// @brief class destructor
  ~MaterialKirchhoff(){}
  /// @fn property
  /// @brief return list of property values
  /// @param [in] aPropertyID 
  /// @return standard vector of strings
  std::vector<std::string> 
  property(const std::string & aPropertyID)
  const 
  {
    auto tEnum = mS2E.get(aPropertyID);
    auto tItr = mProperties.find(tEnum);
    if( tItr == mProperties.end() ){
        return {};
    }
    return tItr->second;
  }
};

/// @class MaterialNeoHookean
/// @brief Material interface for a neo-hookean material model with stored energy potential:
///  \f[
///    \Psi(\mathbf{C})=\frac{1}{2}\lambda(\ln(J))^2 - \mu\ln(J) + \frac{1}{2}\mu(\mbox{trace}(\mathbf{C})-3),
///  \f]
/// where \f$\mathbf{C}\f$ is the right deformation tensor, \f$J=\det(\mathbf{F})\f$, \f$F\f$ 
/// is the deformation gradient, and \f$\lambda\f$ and \f$\mu\f$ are the Lame constants. The 
/// second Piola-Kirchhoff stress tensor is given by:
///  \f[
///    \mathbf{S}=\lambda\ln(J)\mathbf{C}^{-1} + \mu(\mathbf{I}-\mathbf{C}^{-1}),
///  \f]
/// where \f$\mathbf{I}\f$ is the second order identity tensor.
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class MaterialNeoHookean : public Plato::MaterialModel<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions 
  static constexpr int mNumSpatialDims = ElementType::mNumSpatialDims;
  /// @brief map from input string to supported mechanical property
  Plato::mechanical::PropEnum mS2E;
  /// @brief map from mechanical property enum to list of property values in string format
  std::unordered_map<Plato::mechanical::property,std::vector<std::string>> mProperties;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName input material parameter list name
  /// @param [in] aParamList    input material parameter list
  MaterialNeoHookean(
      const std::string            & aMaterialName,
      const Teuchos::ParameterList & aParamList
  )
  {
    // set material input parameter list name
    this->name(aMaterialName);
    // parse youngs modulus
    this->parseScalar("Youngs Modulus", aParamList);
    auto tYoungsModulus = this->getScalarConstant("Youngs Modulus");
    mProperties[mS2E.get("Youngs Modulus")].push_back( std::to_string(tYoungsModulus) );
    // parse poissons ratio
    this->parseScalar("Poissons Ratio", aParamList);
    auto tPoissonsRatio = this->getScalarConstant("Poissons Ratio");
    mProperties[mS2E.get("Poissons Ratio")].push_back( std::to_string(tPoissonsRatio) );
    // compute Lame constants
    auto tMu = tYoungsModulus / ( 2.*(1.+tPoissonsRatio) );
    mProperties[mS2E.get("Lame Mu")].push_back( std::to_string(tMu) );
    auto tLambda = (tPoissonsRatio * tYoungsModulus) / ( (1.+tPoissonsRatio)*(1.-2.*tPoissonsRatio) );
    mProperties[mS2E.get("Lame Lambda")].push_back( std::to_string(tLambda) );
  }

  /// @brief destructor
  ~MaterialNeoHookean(){}
  
  /// @fn property
  /// @brief return list of property values
  /// @param [in] aPropertyID 
  /// @return standard vector of strings
  std::vector<std::string> 
  property(const std::string & aPropertyID)
  const 
  {
    auto tEnum = mS2E.get(aPropertyID);
    auto tItr = mProperties.find(tEnum);
    if( tItr == mProperties.end() ){
        return {};
    }
    return tItr->second;
  }
};

/// @class FactoryNonlinearElasticMaterial
/// @brief factroy for hyperelastic material models
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryNonlinearElasticMaterial
{
private:
    /// @brief const reference to input problem parameter list
    const Teuchos::ParameterList& mParamList;
    /// @brief supported mechanical materials interface
    Plato::mechanical::MaterialEnum mS2E;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  FactoryNonlinearElasticMaterial(
    Teuchos::ParameterList& aParamList
  ) :
    mParamList(aParamList)
  {}
  /// @brief class destructor
  ~FactoryNonlinearElasticMaterial()
  {}
  /// @brief create material model
  /// @param [in] aMaterialName name of input material parameter list
  /// @return shared pointed
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> 
  create(std::string aMaterialName)
  {
    if (!mParamList.isSublist("Material Models"))
    {
      ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
      auto tMaterialModelParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");
      if (!tMaterialModelParamList.isSublist(aMaterialName))
      {
          auto tMsg = std::string("Requested a material model with name ('") + aMaterialName 
                      + "') that is not defined in the input deck";
          ANALYZE_THROWERR(tMsg);
      }
      auto tMaterialParamList = tMaterialModelParamList.sublist(aMaterialName);
      if(tMaterialParamList.isSublist("Kirchhoff")){
        auto tMaterial = std::make_shared<Plato::MaterialKirchhoff<EvaluationType>>
                          (aMaterialName, tMaterialParamList.sublist("Kirchhoff"));
        return tMaterial;
      }
      else
      if(tMaterialParamList.isSublist("Neo-Hookean")){
        auto tMaterial = std::make_shared<Plato::MaterialNeoHookean<EvaluationType>>
                          (aMaterialName, tMaterialParamList.sublist("Neo-Hookean"));
        return tMaterial;
      }
      else{
        mS2E.get("Not Supported"); // throws
        return nullptr;
      }
    }
  }
};

/// @class StressEvaluator
/// @brief base class for stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluator
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;

protected:
  /// @brief contains mesh and model information
  const Plato::SpatialDomain & mSpatialDomain;
  /// @brief output database 
  Plato::DataMap & mDataMap;

public:
  /// @brief base class construtor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  explicit
  StressEvaluator(
      const Plato::SpatialDomain & aSpatialDomain,
            Plato::DataMap       & aDataMap
  ) :
    mSpatialDomain(aSpatialDomain),
    mDataMap(aDataMap)
  {}
  /// @brief base class destructor
  virtual ~StressEvaluator()
  {}

  /// @fn evaluate
  /// @brief pure virtual method: evaluates current density 
  /// @param [in]     aSpatialDomain contains meshed model information
  /// @param [in]     aState         state workset
  /// @param [in]     aControl       control workset
  /// @param [in]     aConfig        configuration workset
  /// @param [in,out] aResult        result workset
  /// @param [in]     aCycle         scalar 
  virtual 
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
      const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
      const Plato::ScalarArray4DT<ResultScalarType>      & aResult,
            Plato::Scalar                                  aCycle = 0.0
  ) const = 0;
};

/// @class KirchhoffSecondPiolaStress
/// @brief Evaluate second Piola-Kirchhoff stress tensor for a Kirchhoff material: \n
///  \f[
///    \mathbf{S}=\lambda\mbox{trace}(\mathbf{E})\mathbf{I} + 2\mu\mathbf{E},
///  \f]
/// where \f$\lambda\f$ and \f$\mu\f$ are the Lame constants, \f$\mathbf{E}\f$ is \n
/// the Green-Lagrange strain tensor, and \f$\mathbf{I}\f$ is the second order \n
/// identity tensor.
/// @tparam EvaluationType 
template<typename EvaluationType>
class KirchhoffSecondPiolaStress
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief Lame constant \f$\mu\f$
  Plato::Scalar mMu;
  /// @brief Lame constant \f$\lambda\f$
  Plato::Scalar mLambda;

public:
  /// @brief class constructor
  /// @param [in] aMaterial material model interface
  KirchhoffSecondPiolaStress(
    const Plato::MaterialModel<EvaluationType> & aMaterial
  )
  {
    mMu     = std::stod(aMaterial.property("lame mu").front());
    mLambda = std::stod(aMaterial.property("lame lambda").front());
  }

  /// @brief class destructor
  ~KirchhoffSecondPiolaStress(){}

  /// @fn operator()()
  /// @brief Compute second Piola-Kirchhoff stress tensor
  /// @param [in]     aStrainTensor Green-Lagrange strain tensor 
  /// @param [in,out] aStressTensor second Piola-Kirchhoff stress tensor
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> & aStrainTensor, 
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> & aStressTensor 
  ) const
  {
    StrainScalarType tTrace = Plato::trace(aStrainTensor);
    for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      aStressTensor(tDimI,tDimI) += mLambda*tTrace;
      for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        aStressTensor(tDimI,tDimJ) += 2.0*mMu*aStrainTensor(tDimI,tDimJ);
      }
    }
  }
};

/// @class StressEvaluatorKirchhoff
/// @brief Evaluate nominal stress for a Kirchhoff material: \n
///  \f[
///    \mathbf{P}_{ij}=S_{ik}F_{jk},\quad i,j,k=1,\dots,N_{dim}
///  \f]
/// where \f$P\f$ is the nominal stress, \f$S\f$ is the second Piola-Kirchhoff \n
/// stress tensor, and \f$F\f$ is the deformation gradient. \n
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluatorKirchhoff : public Plato::StressEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  /// @brief local typename for base class
  using BaseClassType = Plato::StressEvaluator<EvaluationType>;
  /// @brief contains mesh and model information
  using BaseClassType::mSpatialDomain;
  /// @brief output database 
  using BaseClassType::mDataMap;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName  name of material parameter list
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aSpatialDomain contains mesh and model information 
  /// @param [in] aDataMap       output database
  StressEvaluatorKirchhoff(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap) : 
    Plato::StressEvaluator<EvaluationType>(aSpatialDomain,aDataMap)
  {
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterial = tFactory.create(aMaterialName);
  }

  /// @fn evaluate
  /// @brief evaluate stress tensor
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  4D result workset
  /// @param [in]     aCycle   scalar 
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateT>   & aState,
      const Plato::ScalarMultiVectorT<ControlT> & aControl,
      const Plato::ScalarArray3DT<ConfigT>      & aConfig,
      const Plato::ScalarArray4DT<ResultT>      & aResult,
            Plato::Scalar                         aCycle = 0.0
  ) const
  {
    // get integration rule information
    auto tCubPoints = ElementType::getCubPoints();
    auto tNumPoints = ElementType::mNumGaussPoints;
    // compute state gradient
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::GreenLagrangeStrainTensor<EvaluationType> tGreenLagrangeStrainTensor;
    Plato::KirchhoffSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterial);
    // evaluate stress tensor
    auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for("compute nominal stress", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        // compute gradient functions
        ConfigT tVolume(0.0);
        Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> 
          tGradient;
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
        // compute state gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tStateGradient(StrainT(0.));
        tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGradient);
        // compute green-lagrange strain tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tStrainTensor(StrainT(0.));
        tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
        // compute second piola-kirchhoff stress
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
          tStressTensor2PK(ResultT(0.));
        tComputeSecondPiolaKirchhoffStress(tStrainTensor,tStressTensor2PK);
        // compute deformation gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tDefGradient(StrainT(0.));
        tComputeDeformationGradient(tStateGradient,tDefGradient);
        // compute nominal stress
        for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
          for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
            for(Plato::OrdinalType tDimK = 0; tDimK < ElementType::mNumSpatialDims; tDimK++){
              aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) += 
                tStressTensor2PK(tDimI,tDimK)*tDefGradient(tDimJ,tDimK);
            }
          }
        }
    });
  }
};

/// @class NeoHookeanSecondPiolaStress
/// @brief Evaluate second Piola-Kirchhoff stress tensor for a Neo-Hookean material: \n
///  \f[
///    \mathbf{S}=\lambda\ln(J)\mathbf{C}^{-1} + \mu(\mathbf{I}-\mathbf{C}^{-1}),
///  \f]
/// where \f$\lambda\f$ and \f$\mu\f$ are the Lame constants, \f$J=\det(\mathbf{F})\f$, \n 
/// \f$\mathbf{F}\f is the deformation gradient, \f$\mathbf{C}\f$ is the right deformation \n
/// tensor, and \f$\mathbf{I}\f$ is the second order identity tensor. \n
/// @tparam EvaluationType 
template<typename EvaluationType>
class NeoHookeanSecondPiolaStress
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief Lame constant \f$\mu\f$
  Plato::Scalar mMu;
  /// @brief Lame constant \f$\lambda\f$
  Plato::Scalar mLambda;
  /// @brief computes right deformation tensor 
  Plato::RightDeformationTensor<EvaluationType> mComputeRightDeformationTensor;

public:
  /// @brief class constructor
  /// @param [in] aMaterial material model interface
  NeoHookeanSecondPiolaStress(
    const Plato::MaterialModel<EvaluationType> & aMaterial
  )
  {
    mMu     = std::stod(aMaterial.property("lame mu").front());
    mLambda = std::stod(aMaterial.property("lame lambda").front());
  }

  /// @brief class destructor
  ~NeoHookeanSecondPiolaStress(){}

  /// @fn operator()()
  /// @brief Compute second Piola-Kirchhoff stress tensor
  /// @param [in]     aDefGradient  deformation gradient  
  /// @param [in,out] aStressTensor second Piola-Kirchhoff stress tensor
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> & aDefGradient, 
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> & aStressTensor 
  ) const
  {
    // apply transpose to deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> tDefGradientT = 
      Plato::transpose(aDefGradient);
    // compute cauchy-green deformation tensor
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tRightDeformationTensor(StrainScalarType(0.));
    mComputeRightDeformationTensor(tDefGradientT,aDefGradient,tRightDeformationTensor);
    // compute determinant of deformation gradient
    StrainScalarType tDetDefGrad = Plato::determinant(aDefGradient);
    // invert right deformation tensor
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tInverseRightDeformationTensor = Plato::invert(tRightDeformationTensor);
    // compute stress tensor
    for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      aStressTensor(tDimI,tDimI) += mMu;
      for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        aStressTensor(tDimI,tDimJ) += mLambda*log(tDetDefGrad)*tInverseRightDeformationTensor(tDimI,tDimJ);
        aStressTensor(tDimI,tDimJ) -= mMu*tInverseRightDeformationTensor(tDimI,tDimJ);
      }
    }
  }
};

/// @class StressEvaluatorNeoHookean
/// @brief Evaluate nominal stress for a Neo-Hookean material: \n
///  \f[
///    \mathbf{P}_{ij}=S_{ik}F_{jk},\quad i,j,k=1,\dots,N_{dim}
///  \f]
/// where \f$P\f$ is the nominal stress, \f$S\f$ is the second Piola-Kirchhoff \n
/// stress tensor, and \f$F\f$ is the deformation gradient. \n
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class StressEvaluatorNeoHookean : public Plato::StressEvaluator<EvaluationType>
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  /// @brief local typename for base class
  using BaseClassType = Plato::StressEvaluator<EvaluationType>;
  /// @brief contains mesh and model information
  using BaseClassType::mSpatialDomain;
  /// @brief output database 
  using BaseClassType::mDataMap;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName  name of material parameter list
  /// @param [in] aParamList     input problem parameters
  /// @param [in] aSpatialDomain contains mesh and model information 
  /// @param [in] aDataMap       output database
  StressEvaluatorNeoHookean(
    const std::string            & aMaterialName,
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap) : 
    Plato::StressEvaluator<EvaluationType>(aSpatialDomain,aDataMap)
  {
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterial = tFactory.create(aMaterialName);
  }

  /// @brief class destructor
  ~StressEvaluatorNeoHookean(){}

  /// @fn evaluate
  /// @brief evaluate stress tensor
  /// @param [in]     aState   2D state workset
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  4D result workset
  /// @param [in]     aCycle   scalar 
  void 
  evaluate(
      const Plato::ScalarMultiVectorT<StateT>   & aState,
      const Plato::ScalarMultiVectorT<ControlT> & aControl,
      const Plato::ScalarArray3DT<ConfigT>      & aConfig,
      const Plato::ScalarArray4DT<ResultT>      & aResult,
      Plato::Scalar                               aCycle = 0.0
  ) const
  {
    // get integration rule information
    auto tCubPoints = ElementType::getCubPoints();
    auto tNumPoints = ElementType::mNumGaussPoints;
    // compute state gradient
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::NeoHookeanSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterial);
    // evaluate stress tensor
    auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for("compute nominal stress", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        // compute gradient functions
        ConfigT tVolume(0.0);
        Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
        // compute state gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
        tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGradient);
        // compute deformation gradient 
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> tDefGradient(StrainT(0.));
        tComputeDeformationGradient(tStateGradient,tDefGradient);
        // compute second-piola kirchhoff stress tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
          tStressTensor2PK(ResultT(0.));
        tComputeSecondPiolaKirchhoffStress(tDefGradient,tStressTensor2PK);
        // compute nominal stress
        for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
          for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
            for(Plato::OrdinalType tDimK = 0; tDimK < ElementType::mNumSpatialDims; tDimK++){
              aResult(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) += 
                tStressTensor2PK(tDimI,tDimK)*tDefGradient(tDimJ,tDimK);
            }
          }
        }
    });
  }
};

/// @class FactoryStressEvaluator
/// @brief Factory of mechanical stress evaluators
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class FactoryStressEvaluator
{
private:
  /// @brief name of input material parameter list
  std::string mMaterialName; 
  /// @brief supported mechanical materials interface
  Plato::mechanical::MaterialEnum mS2E;

public:
  /// @brief class constructor
  /// @param [in] aParamList input problem parameters
  FactoryStressEvaluator(
    const std::string & aMaterialName
  ) :
    mMaterialName(aMaterialName)
  {}

  /// @brief class destructor
  ~FactoryStressEvaluator(){}
  
  /// @fn create 
  /// @brief create stress evaluator
  /// @param [in] aMaterialName  name of input material parameter list
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @return shared pointer
  std::shared_ptr<Plato::StressEvaluator<EvaluationType>> 
  create(
          Teuchos::ParameterList & aParamList,
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap
  )
  {
    if (!aParamList.isSublist("Material Models"))
    {
      ANALYZE_THROWERR("ERROR: 'Material Models' parameter list not found! Returning 'nullptr'");
    }
    else
    {
      auto tMaterialModelParamList = aParamList.get<Teuchos::ParameterList>("Material Models");
      if (!tMaterialModelParamList.isSublist(mMaterialName))
      {
          auto tMsg = std::string("Requested a material model with name ('") + mMaterialName 
            + "') that is not defined in the input deck";
          ANALYZE_THROWERR(tMsg);
      }
      auto tMaterialParamList = tMaterialModelParamList.sublist(mMaterialName);
      if(tMaterialParamList.isSublist("Kirchhoff")){
        auto tStressEvaluator = std::make_shared<Plato::StressEvaluatorKirchhoff<EvaluationType>>
                                (mMaterialName,aParamList,aSpatialDomain,aDataMap);
        return tStressEvaluator;
      }
      else
      if(tMaterialParamList.isSublist("Neo-Hookean")){
        auto tStressEvaluator = std::make_shared<Plato::StressEvaluatorNeoHookean<EvaluationType>>
                                (mMaterialName,aParamList,aSpatialDomain,aDataMap);
        return tStressEvaluator;
      }
      else{
        mS2E.get("Not Supported"); // throws
        return nullptr;
      }
    }
  }
};

template<typename EvaluationType>
class ResidualElastostaticTotalLagrangian : 
  public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief number of degrees of freedom per node
  static constexpr auto mNumDofsPerNode = ElementType::mNumDofsPerNode;
  /// @brief number of degrees of freedom per cell
  static constexpr auto mNumDofsPerCell = ElementType::mNumDofsPerCell;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;
  /// @brief number of integration points per cell
  static constexpr auto mNumGaussPoints = ElementType::mNumGaussPoints;
  /// @brief local typename for base class
  using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;
  /// @brief contains mesh and model information
  using FunctionBaseType::mSpatialDomain;
  /// @brief output database
  using FunctionBaseType::mDataMap;
  /// @brief contains degrees of freedom names 
  using FunctionBaseType::mDofNames;
  /// @brief scalar types associated with the automatic differentation evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
  /// @brief stress evaluator
  std::shared_ptr<Plato::StressEvaluator<EvaluationType>> mStressEvaluator;
  /// @brief natural boundary conditions evaluator
  std::shared_ptr<Plato::NaturalBCs<ElementType>> mNaturalBCs;
  /// @brief body loads evaluator
  std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
  /// @brief output plot table, contains requested output quantities of interests
  std::vector<std::string> mPlotTable;

public:
  /// @brief class constructor
  /// @param [in] aSpatialDomain contains mesh and model information
  /// @param [in] aDataMap       output database
  /// @param [in] aParamList     input problem parameters
  ResidualElastostaticTotalLagrangian(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList
  ) :
    FunctionBaseType(aSpatialDomain, aDataMap),
    mStressEvaluator(nullptr),
    mNaturalBCs     (nullptr),
    mBodyLoads      (nullptr)
  {
    // obligatory: define dof names in order
    //
    mDofNames.push_back("displacement X");
    if(mNumSpatialDims > 1) mDofNames.push_back("displacement Y");
    if(mNumSpatialDims > 2) mDofNames.push_back("displacement Z");
    // initialize member data
    //
    this->initialize(aParamList);
  }

  /// @brief class destructor
  ~ResidualElastostaticTotalLagrangian(){}

  /// @fn getSolutionStateOutputData
  /// @brief post-process state solution for output
  /// @param [in,out] aSolutions solution database
  /// @return solution database
  Plato::Solutions
  getSolutionStateOutputData(
    const Plato::Solutions & aSolutions
  ) const
  {
    // No scaling, addition, or removal of data necessary for this physics.
    return aSolutions;
  }

  /// @fn evaluate
  /// @brief evaluate internal forces
  /// @param [in]     aState   2D state workset 
  /// @param [in]     aControl 2D control workset
  /// @param [in]     aConfig  3D configuration workset
  /// @param [in,out] aResult  2D result workset
  /// @param [in]     aCycle   scalar 
  void
  evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT    <ConfigScalarType>  & aConfig,
          Plato::ScalarMultiVectorT<ResultScalarType>  & aResult,
          Plato::Scalar                                  aCycle
  ) const
  {
    // evaluate stresses
    Plato::OrdinalType tNumCells = mSpatialDomain.numCells();
    Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
    Plato::ScalarArray4DT<ResultScalarType> 
      tNominalStress("nominal stress",tNumCells,tNumGaussPoints,mNumSpatialDims,mNumSpatialDims);
    mStressEvaluator->evaluate(aState,aControl,aConfig,tNominalStress,aCycle);
    // get integration rule data
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    // evaluate internal forces
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Kokkos::parallel_for("compute internal forces", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,mNumGaussPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        // compute gradient of interpolation functions
        ConfigScalarType tVolume(0.0);
        auto tCubPoint = tCubPoints(iGpOrdinal);
        Plato::Matrix<mNumNodesPerCell,mNumSpatialDims,ConfigScalarType> tGradient;
        tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
        // apply integration point weight to element volume
        tVolume *= tCubWeights(iGpOrdinal);
        // apply divergence operator to stress tensor
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++){
          for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
            Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumSpatialDims + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
              ResultScalarType tVal = tNominalStress(iCellOrdinal,iGpOrdinal,tDimI,tDimJ) 
                * tGradient(tNodeIndex,tDimJ) * tVolume;
              Kokkos::atomic_add( &aResult(iCellOrdinal,tLocalOrdinal),tVal );
            }
          }
        }
    });
    // evaluate body forces
    if( mBodyLoads != nullptr )
    {
      mBodyLoads->get( mSpatialDomain,aState,aControl,aConfig,aResult,-1.0 );
    }
  }

  /// @fn evaluate_boundary
  /// @brief evaluate boundary forces
  /// @param [in]     aSpatialModel contains mesh and model information
  /// @param [in]     aState        2D state workset 
  /// @param [in]     aControl      2D control workse
  /// @param [in]     aConfig       3D configuration 
  /// @param [in,out] aResult       2D result workset
  /// @param [in]     aCycle        scalar
  void
  evaluate_boundary(
    const Plato::SpatialModel                           & aSpatialModel,
    const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
    const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
    const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
          Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
          Plato::Scalar                                   aCycle
  ) const
  {
    if( mNaturalBCs != nullptr )
    {
      mNaturalBCs->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
    }
  }

private:
  /// @fn initialize
  /// @brief initialize member data
  /// @param [in] aParamList input problem parameters
  void initialize(
    Teuchos::ParameterList & aParamList
  )
  {
    // create material model and get stiffness
    //
    Plato::FactoryStressEvaluator<EvaluationType> tStressEvaluatorFactory(mSpatialDomain.getMaterialName());
    mStressEvaluator = tStressEvaluatorFactory.create(aParamList,mSpatialDomain,mDataMap);
    // parse body loads
    // 
    if(aParamList.isSublist("Body Loads"))
    {
      mBodyLoads = 
        std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>(aParamList.sublist("Body Loads"));
    }
    // parse boundary Conditions
    // 
    if(aParamList.isSublist("Natural Boundary Conditions"))
    {
      mNaturalBCs = 
        std::make_shared<Plato::NaturalBCs<ElementType>>(aParamList.sublist("Natural Boundary Conditions"));
    }
    // parse plot table
    //
    auto tResidualParams = aParamList.sublist("Output");
    if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
    {
      mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }
  }
};

template<typename EvaluationType>
class CriterionKirchhoffElasticEnergyPotential :
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief local typename for base class
  using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
  /// @brief number of spatial dimensions
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;

  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  CriterionKirchhoffElasticEnergyPotential(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
    const std::string            & aFuncName
  ) :
    FunctionBaseType(aSpatialDomain,aDataMap,aParamList,aFuncName)
  {
    std::string tMaterialName = mSpatialDomain.getMaterialName();
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterial = tFactory.create(tMaterialName);
  }

  ~CriterionKirchhoffElasticEnergyPotential(){}

  void 
  evaluate_conditional(
      const Plato::ScalarMultiVectorT <StateT>   & aState,
      const Plato::ScalarMultiVectorT <ControlT> & aControl,
      const Plato::ScalarArray3DT     <ConfigT>  & aConfig,
            Plato::ScalarVectorT      <ResultT>  & aResult,
            Plato::Scalar                          aCycle
  ) const
  {
    // get integration rule information
    auto tNumPoints  = ElementType::mNumGaussPoints;
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    // compute state gradient
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::GreenLagrangeStrainTensor<EvaluationType> tGreenLagrangeStrainTensor;
    Plato::KirchhoffSecondPiolaStress<EvaluationType> tComputeSecondPiolaKirchhoffStress(*mMaterial);
    // evaluate stress tensor
    auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for("evaluate strain energy potential", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        // compute gradient functions
        ConfigT tVolume(0.0);
        Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> 
          tGradient;
        auto tCubPoint = tCubPoints(iGpOrdinal);
        tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
        // compute state gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tStateGradient(StrainT(0.));
        tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGradient);
        // compute green-lagrange strain tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tStrainTensor(StrainT(0.));
        tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
        // compute second piola-kirchhoff stress
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultT> 
          tStressTensor2PK(ResultT(0.));
        tComputeSecondPiolaKirchhoffStress(tStrainTensor,tStressTensor2PK);
        // apply integration point weight to element volume
        tVolume *= tCubWeights(iGpOrdinal);
        // evaluate elastic strain energy potential 
        ResultT tValue(0.0);
        for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
          for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
            tValue += tStrainTensor(tDimI,tDimJ) * tStressTensor2PK(tDimI,tDimJ) * tVolume;
          }
        }
        Kokkos::atomic_add(&aResult(iCellOrdinal), tValue);
    });
  }
};

template<typename EvaluationType>
class CriterionNeoHookeanElasticEnergyPotential :
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell;
  /// @brief local typename for base class
  using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
  /// @brief number of spatial dimensions
  using FunctionBaseType::mSpatialDomain;
  using FunctionBaseType::mDataMap;

  using StateT   = typename EvaluationType::StateScalarType;
  using ConfigT  = typename EvaluationType::ConfigScalarType;
  using ResultT  = typename EvaluationType::ResultScalarType;
  using ControlT = typename EvaluationType::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  /// @brief material constitutive model interface
  std::shared_ptr<Plato::MaterialModel<EvaluationType>> mMaterial;

public:
  CriterionNeoHookeanElasticEnergyPotential(
    const Plato::SpatialDomain   & aSpatialDomain,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aParamList,
    const std::string            & aFuncName
  ) :
    FunctionBaseType(aSpatialDomain,aDataMap,aParamList,aFuncName)
  {
    std::string tMaterialName = mSpatialDomain.getMaterialName();
    Plato::FactoryNonlinearElasticMaterial<EvaluationType> tFactory(aParamList);
    mMaterial = tFactory.create(tMaterialName);
  }

  ~CriterionNeoHookeanElasticEnergyPotential(){}

  void 
  evaluate_conditional(
      const Plato::ScalarMultiVectorT <StateT>   & aState,
      const Plato::ScalarMultiVectorT <ControlT> & aControl,
      const Plato::ScalarArray3DT     <ConfigT>  & aConfig,
            Plato::ScalarVectorT      <ResultT>  & aResult,
            Plato::Scalar                          aCycle
  ) const
  {
    // get material properties
    auto tMu     = std::stod(mMaterial->property("lame mu").front());
    auto tLambda = std::stod(mMaterial->property("lame lambda").front());
    // get integration rule information
    auto tNumPoints  = ElementType::mNumGaussPoints;
    auto tCubPoints  = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    // compute state gradient
    Plato::StateGradient<EvaluationType> tComputeStateGradient;
    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::DeformationGradient<EvaluationType> tComputeDeformationGradient;
    Plato::RightDeformationTensor<EvaluationType> tComputeRightDeformationTensor;
    // evaluate stress tensor
    auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for("evaluate strain energy potential", 
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
        // compute gradient functions
        ConfigT tVolume(0.0);
        auto tCubPoint = tCubPoints(iGpOrdinal);
        Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
        tComputeGradient(iCellOrdinal,tCubPoint,aConfig,tGradient,tVolume);
        // compute state gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tStateGradient(StrainT(0.));
        tComputeStateGradient(iCellOrdinal,aState,tGradient,tStateGradient);
        // compute green-lagrange strain tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tDeformationGradient(StrainT(0.));
        tComputeDeformationGradient(tStateGradient,tDeformationGradient);
        // apply transpose to deformation gradient
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tDeformationGradientT = Plato::transpose(tDeformationGradient);
        // compute right deformation tensor
        Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainT> 
          tRightDeformationTensor(StrainT(0.));
        tComputeRightDeformationTensor(tDeformationGradientT,tDeformationGradient,tRightDeformationTensor);
        // compute trace of right deformation tensor
        StrainT tTrace = Plato::trace(tRightDeformationTensor);
        // compute determinant of deformation gradient
        StrainT tDetDeformationGradient = Plato::determinant(tDeformationGradient);
        // evaluate elastic strain energy potential
        StrainT tLogDetF = log(tDetDeformationGradient);
        ResultT tValue = 0.5*tLambda*tLogDetF*tLogDetF - tMu*tLogDetF + 0.5*tMu*(tTrace-3.0);
        Kokkos::atomic_add(&aResult(iCellOrdinal), tValue);
    });
  }
};

}

namespace ElastostaticTotalLagrangianTests
{

Teuchos::RCP<Teuchos::ParameterList> tGenericParamList = Teuchos::getParametersFromXmlString(
"<ParameterList name='Plato Problem'>                                                                  \n"
  "<ParameterList name='Spatial Model'>                                                                \n"
    "<ParameterList name='Domains'>                                                                    \n"
      "<ParameterList name='Design Volume'>                                                            \n"
        "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
        "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
      "</ParameterList>                                                                                \n"
    "</ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
  "<ParameterList name='Material Models'>                                                              \n"
    "<ParameterList name='Mystic'>                                                                     \n"
      "<ParameterList name='Kirchhoff'>                                                                \n"
        "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
        "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>                                 \n"
      "</ParameterList>                                                                                \n"
    "</ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
  "<ParameterList name='Criteria'>                                                                     \n"
  "  <ParameterList name='Objective'>                                                                  \n"
  "    <Parameter name='Type' type='string' value='Weighted Sum'/>                                     \n"
  "    <Parameter name='Functions' type='Array(string)' value='{My Strain Energy}'/>                   \n"
  "    <Parameter name='Weights' type='Array(double)' value='{1.0}'/>                                  \n"
  "  </ParameterList>                                                                                  \n"
  "  <ParameterList name='My Strain Energy'>                                                           \n"
  "    <Parameter name='Type'                 type='string' value='Scalar Function'/>                  \n"
  "    <Parameter name='Scalar Function Type' type='string' value='Strain Energy Potential'/>          \n"
  "  </ParameterList>                                                                                  \n"
  "</ParameterList>                                                                                    \n"
"</ParameterList>                                                                                      \n"
); 

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, tComputeStateGradient )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // copy results
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tStateGradient(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.4,0.2,0.4,0.2}, {0.4,0.2,0.4,0.2} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, DeformationGradient )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradient(StrainT(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tDefGradient(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.4,0.2,0.4,1.2}, {1.4,0.2,0.4,1.2} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, RightDeformationTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Plato::RightDeformationTensor<Residual> tComputeRightDeformationTensor;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute deformation gradient 
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradient(StrainT(0.));
      tComputeDeformationGradient(tStateGradient,tDefGradient);
      // apply transpose to deformation gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefGradT
        = Plato::transpose(tDefGradient);
      // compute cauchy-green deformation tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tDefTensor(StrainT(0.));
      tComputeRightDeformationTensor(tDefGradT,tDefGradient,tDefTensor);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tDefTensor(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {2.12,0.76,0.76,1.48}, {2.12,0.76,0.76,1.48} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, GreenLagrangeStrainTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // results workset
  Plato::ScalarArray3DT<ResultT> tResultsWS("results",tNumCells,tSpaceDim,tSpaceDim);
  // get integration rule information
  auto tCubPoints  = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints  = tCubWeights.size();
  // compute state gradient
  Plato::StateGradient<Residual> tComputeStateGradient;
  Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
  Plato::DeformationGradient<Residual> tComputeDeformationGradient;
  Plato::RightDeformationTensor<Residual> tComputeRightDeformationTensor;
  Plato::GreenLagrangeStrainTensor<Residual> tGreenLagrangeStrainTensor;
  Kokkos::parallel_for("compute state gradient", 
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {tNumCells,tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
      // compute gradient functions
      ConfigT tVolume(0.0);
      Plato::Matrix<ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims,ConfigT> tGradient;
      auto tCubPoint = tCubPoints(iGpOrdinal);
      tComputeGradient(iCellOrdinal,tCubPoint,tConfigWS,tGradient,tVolume);
      // compute state gradient
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStateGradient(StrainT(0.));
      tComputeStateGradient(iCellOrdinal,tStateWS,tGradient,tStateGradient);
      // compute green-lagrange strain tensor
      Plato::Matrix<ElementType::mNumSpatialDims ,ElementType::mNumSpatialDims,StrainT> tStrainTensor(StrainT(0.));
      tGreenLagrangeStrainTensor(tStateGradient,tStrainTensor);
      // copy result
      for( Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
        for( Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
          tResultsWS(iCellOrdinal,tDimI,tDimJ) = tStrainTensor(tDimI,tDimJ);
        }
      }
  });
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.56,0.38,0.38,0.24}, {0.56,0.38,0.38,0.24} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, StressEvaluatorKirchhoffTensor )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarArray4DT<ResultT> tResultsWS("stress",tNumCells,tNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::StressEvaluatorKirchhoff<Residual> tStressEvaluator("Mystic",*tGenericParamList,tOnlyDomainDefined,tDataMap);
  tStressEvaluator.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {1.60493827,0.78024691,0.56790123,1.15555556}, {1.60493827,0.78024691,0.56790123,1.15555556} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,0,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, StressEvaluatorNeoHookeanTensor )
{
  // create parameter list
  Teuchos::RCP<Teuchos::ParameterList> tParamList = Teuchos::getParametersFromXmlString(
  "<ParameterList name='Plato Problem'>                                                                  \n"
    "<ParameterList name='Spatial Model'>                                                                \n"
      "<ParameterList name='Domains'>                                                                    \n"
        "<ParameterList name='Design Volume'>                                                            \n"
          "<Parameter name='Element Block' type='string' value='body'/>                                  \n"
          "<Parameter name='Material Model' type='string' value='Mystic'/>                               \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
    "<ParameterList name='Material Models'>                                                              \n"
      "<ParameterList name='Mystic'>                                                                     \n"
        "<ParameterList name='Neo-Hookean'>                                                              \n"
          "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>                                \n"
          "<Parameter  name='Youngs Modulus' type='double' value='1.0'/>                                 \n"
        "</ParameterList>                                                                                \n"
      "</ParameterList>                                                                                  \n"
    "</ParameterList>                                                                                    \n"
  "</ParameterList>                                                                                      \n"
  );
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarArray4DT<ResultT> tResultsWS("stress",tNumCells,tNumGaussPoints,tSpaceDim,tSpaceDim);
  Plato::StressEvaluatorNeoHookean<Residual> 
    tStressEvaluator("Mystic",*tParamList,tOnlyDomainDefined,tDataMap);
  tStressEvaluator.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { {0.54537272,0.14367245,0.06512267,0.47577435}, {0.54537272,0.14367245,0.06512267,0.47577435} };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDimI = 0; tDimI < tSpaceDim; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < tSpaceDim; tDimJ++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDimI*tSpaceDim+tDimJ],tHostResultsWS(tCell,0,tDimI,tDimJ),tTolerance);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, Residual )
{
  // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create results workset
  Plato::OrdinalType tNumGaussPoints = ElementType::mNumGaussPoints;
  Plato::ScalarMultiVectorT<ResultT> tResultsWS("residual",tNumCells,tDofsPerCell);
  // evaluate residual
  Plato::ResidualElastostaticTotalLagrangian<Residual> 
    tResidual(tOnlyDomainDefined,tDataMap,*tGenericParamList);
  tResidual.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS,0.0);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<std::vector<Plato::Scalar>> tGold = 
    { 
      {-0.802469,-0.28395 ,0.412346,-0.293827,0.390123 ,0.577778}, 
      {-0.390123,-0.577778,0.802469,0.28395  ,-0.412346,0.293827} 
    };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    for(Plato::OrdinalType tDof = 0; tDof < tDofsPerCell; tDof++){
        TEST_FLOATING_EQUALITY(tGold[tCell][tDof],tHostResultsWS(tCell,tDof),tTolerance);
    }
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, CriterionKirchhoffElasticEnergyPotential )
{
 // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create results workset
  Plato::ScalarVectorT<ResultT> tResultsWS("residual",tNumCells);
  // create criterion
  Plato::CriterionKirchhoffElasticEnergyPotential<Residual> 
    tCriterion(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Strain Energy");
  tCriterion.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<Plato::Scalar> tGold = { 0.52098765432098735, 0.52098765432098735 };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    TEST_FLOATING_EQUALITY(tGold[tCell],tHostResultsWS(tCell),tTolerance);
  }
}

TEUCHOS_UNIT_TEST( ElastostaticTotalLagrangianTests, CriterionNeoHookeanElasticEnergyPotential )
{
 // create mesh
  constexpr Plato::OrdinalType tSpaceDim = 2;
  constexpr Plato::OrdinalType tMeshWidth = 1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
  using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;  
  // create output database and spatial model
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tGenericParamList, tDataMap);
  auto tOnlyDomainDefined = tSpatialModel.Domains.front();
  //set ad-types
  using Residual = typename Plato::Elliptic::Evaluation<Plato::MechanicsElement<Plato::Tri3>>::Residual;
  using StateT   = typename Residual::StateScalarType;
  using ConfigT  = typename Residual::ConfigScalarType;
  using ResultT  = typename Residual::ResultScalarType;
  using ControlT = typename Residual::ControlScalarType;
  using StrainT  = typename Plato::fad_type_t<ElementType,StateT,ConfigT>;
  // create configuration workset
  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  const Plato::OrdinalType tNumCells = tMesh->NumElements();
  TEST_EQUALITY(2,tNumCells);
  constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
  Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
  tWorksetBase.worksetConfig(tConfigWS);
  // create state workset
  const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
  const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
  Plato::ScalarVector tState("States", tNumDofs);
  Plato::blas1::fill(0.1, tState);
  Kokkos::parallel_for("fill state",
    Kokkos::RangePolicy<>(0, tNumDofs), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); });
  constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
  Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
  tWorksetBase.worksetState(tState, tStateWS);
  // create control workset
  Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset",tNumCells,tNodesPerCell);
  Plato::ScalarVector tControl("Controls", tNumVerts);
  Plato::blas1::fill(1.0, tControl);
  tWorksetBase.worksetControl(tControl, tControlWS);
  // create results workset
  Plato::ScalarVectorT<ResultT> tResultsWS("residual",tNumCells);
  // create criterion
  Plato::CriterionNeoHookeanElasticEnergyPotential<Residual> 
    tCriterion(tOnlyDomainDefined,tDataMap,*tGenericParamList,"My Strain Energy");
  tCriterion.evaluate(tStateWS,tControlWS,tConfigWS,tResultsWS);
  // test gold values
  constexpr Plato::Scalar tTolerance = 1e-4;
  std::vector<Plato::Scalar> tGold = { 0.032487784262637334, 0.032487784262637334 };
  auto tHostResultsWS = Kokkos::create_mirror(tResultsWS);
  Kokkos::deep_copy(tHostResultsWS, tResultsWS);
  for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++){
    TEST_FLOATING_EQUALITY(tGold[tCell],tHostResultsWS(tCell),tTolerance);
  }
}

}