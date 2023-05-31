/*
 * MaterialNeoHookean_decl.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include <vector>

#include <Teuchos_ParameterList.hpp>

#include "MaterialModel.hpp"
#include "elliptic/mechanical/SupportedParamOptions.hpp"

namespace Plato
{

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
  );

  /// @brief destructor
  ~MaterialNeoHookean(){}
  
  /// @fn property
  /// @brief return list of property values
  /// @param [in] aPropertyID 
  /// @return standard vector of strings
  std::vector<std::string> 
  property(const std::string & aPropertyID)
  const;    
};

}