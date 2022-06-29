#pragma once

#ifdef PLATO_HEX_ELEMENT

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

