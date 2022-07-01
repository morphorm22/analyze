#pragma once

#include "Tet4.hpp"
#include "Tet10.hpp"
#include "Tri3.hpp"

#ifdef PLATO_HEX_ELEMENTS

#include "Hex8.hpp"
#include "Quad4.hpp"
#include "Hex27.hpp"

#define PLATO_ELEMENT_DEF(C, T) \
template class C<T<Plato::Tet4>>; \
template class C<T<Plato::Tet10>>; \
template class C<T<Plato::Tri3>>; \
template class C<T<Plato::Hex8>>; \
template class C<T<Plato::Quad4>>; \
template class C<T<Plato::Hex27>>;

#define PLATO_ELEMENT_DEC(C, T) \
extern template class C<T<Plato::Tet4>>; \
extern template class C<T<Plato::Tet10>>; \
extern template class C<T<Plato::Tri3>>; \
extern template class C<T<Plato::Hex8>>; \
extern template class C<T<Plato::Quad4>>; \
extern template class C<T<Plato::Hex27>>;

#else

#define PLATO_ELEMENT_DEF(C, T) \
template class C<T<Plato::Tet4>>; \
template class C<T<Plato::Tet10>>; \
template class C<T<Plato::Tri3>>;

#define PLATO_ELEMENT_DEC(C, T) \
extern template class C<T<Plato::Tet4>>; \
extern template class C<T<Plato::Tet10>>; \
extern template class C<T<Plato::Tri3>>;

#endif

