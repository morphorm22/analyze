#pragma once

#define PLATO_ELLIPTIC_UPLAG_EXPL_DEF(C, T, D) \
template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::Hatching::GradientCTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::Hatching::GradientCTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::Hatching::GradientCTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_ELLIPTIC_UPLAG_EXPL_DEC(C, T, D) \
extern template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::Hatching::GradientCTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::Hatching::GradientCTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::Hatching::GradientCTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, Plato::Heaviside >;


#define PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(C, T, D) \
extern template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, T<D> >; \
extern template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, T<D> >; \
extern template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, T<D> >; \
extern template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, T<D> >;

#define PLATO_ELLIPTIC_UPLAG_EXPL_DEF2(C, T, D) \
template class C<Plato::Elliptic::Hatching::ResidualTypes<T<D>>, T<D> >; \
template class C<Plato::Elliptic::Hatching::JacobianTypes<T<D>>, T<D> >; \
template class C<Plato::Elliptic::Hatching::GradientXTypes<T<D>>, T<D> >; \
template class C<Plato::Elliptic::Hatching::GradientZTypes<T<D>>, T<D> >;

