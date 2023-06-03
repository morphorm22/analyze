/*
 * MetaData.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <memory>

namespace Plato
{

/// @class MetaDataBase
/// @brief abstraction layer for workeset data
class MetaDataBase
{
public:
  /// @brief class destructor
  virtual ~MetaDataBase() = 0;
};
inline MetaDataBase::~MetaDataBase(){}
// class MetaDataBase

/// @class MetaData
/// @brief provides an abstraction layer for workset data
/// @tparam Type data type
template<class Type>
class MetaData : public MetaDataBase
{
public:
  /// @brief class constructor
  /// @param [in] aData data
  explicit MetaData(const Type &aData) : mData(aData) {}

  /// @brief class constructor
  MetaData() {}
  
  /// @brief abstract metadata
  Type mData; 
};
// class MetaData

/// @fn unpack
/// @brief unpack data from metadata abstraction
/// @tparam Type data type
/// @param aInput shared pointer to metadata abstraction
/// @return return reference to data with typename Type
template<class Type>
inline Type 
unpack(
  const std::shared_ptr<Plato::MetaDataBase> & aInput
)
{
  return (dynamic_cast<Plato::MetaData<Type>&>(aInput.operator*()).mData);
}

} // namespace Plato
