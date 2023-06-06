/*
 * WorksetTests.cpp
 *
 *  Created on: June 6, 2023
 */

namespace Plato
{

/// @brief workset builder base class. derived class builds worksets relevant to 
/// the physics and partial differential equation of interest
class WorksetBuilderBase
{
public:
  /// @brief build data worksets
  /// @param [in]     aDomain   contains mesh and model information
  /// @param [in]     aDatabase contains control, state, and other data needed to evaluate residual and criteria
  /// @param [in,out] aWorkSets workset database
  virtual 
  void build(
    const Plato::SpatialDomain & aDomain,
    const Plato::Database      & aDatabase,
          Plato::WorkSets      & aWorkSets
  ) const = 0;

  /// @brief build data worksets
  /// @param [in]     aNumCells local number of cells/elements
  /// @param [in]     aDatabase contains control, state, and other data needed to evaluate residual and criteria
  /// @param [in,out] aWorkSets workset database
  virtual 
  void build(
    const Plato::OrdinalType & aNumCells,
    const Plato::Database    & aDatabase,
          Plato::WorkSets    & aWorkSets
  ) const = 0;
};

} // namespace Plato
