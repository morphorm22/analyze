/*
 * CurrentDensityModel.hpp
 *
 *  Created on: May 24, 2023
 */

#pragma once

#include "PlatoTypes.hpp"

namespace Plato
{

/// @class CurrentDensityModel
/// @brief base class for current density models
/// @tparam EvaluationType   automatic differentiation evaluation type, which sets scalar types
/// @tparam OutputScalarType output scalar type 
template<typename EvaluationType, 
         typename OutputScalarType>
class CurrentDensityModel
{
private:
    /// @brief topological element type
    using ElementType = typename EvaluationType::ElementType;
    /// @brief state scalar type
    using StateScalarType = typename EvaluationType::StateScalarType;

public:
    /// @brief class constructor
    CurrentDensityModel(){}
    /// @brief class destructor 
    virtual ~CurrentDensityModel(){}

    /// @fn evaluate
    /// @brief pure virtual method, evaluates current density model
    /// @param [in] aState  2D state workset
    /// @param [in] aResult 2D output workset
    virtual 
    void 
    evaluate(
      const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
      const Plato::ScalarMultiVectorT <OutputScalarType>  & aResult
    ) const = 0;
};

}
// namespace Plato