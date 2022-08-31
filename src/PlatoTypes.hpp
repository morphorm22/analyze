/*
 * PlatoTypes.hpp
 *
 *  Created on: Jul 12, 2018
 */

#ifndef SRC_PLATO_PLATOTYPES_HPP_
#define SRC_PLATO_PLATOTYPES_HPP_

#include <Kokkos_Core.hpp>

namespace Plato
{

using Scalar = double;
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
  using OrdinalType = long long int;
#else
  using OrdinalType = int;
#endif
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

#if defined(KOKKOS_ENABLE_CUDA)
  using UVMSpace = Kokkos::CudaUVMSpace;
#else
  using UVMSpace = ExecSpace::memory_space;
#endif

#define MAX_ARRAY_LENGTH 128

using Layout = Kokkos::LayoutRight;

// Map structure - used with Kokkos so char strings so to be compatable.
template< typename KEY_TYPE, typename VALUE_TYPE > struct _Map {
  KEY_TYPE key;
  VALUE_TYPE value;
};

template< typename KEY_TYPE, typename VALUE_TYPE >
using Map = _Map< KEY_TYPE, VALUE_TYPE>;

using VariableMap = Map< Plato::OrdinalType, char[MAX_ARRAY_LENGTH] >;

} // namespace Plato

#endif /* SRC_PLATO_PLATOTYPES_HPP_ */
