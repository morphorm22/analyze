#pragma once

#include <type_traits>
#include <iterator>

#include <hpc_array_traits.hpp>
#include <hpc_matrix.hpp>

namespace hpc {

namespace impl {

template <class T, layout L>
class array_vector_reference {
  public:
  using array_value_type = typename ::hpc::array_traits<T>::value_type;
  using iterator_type = ::hpc::impl::inner_iterator<::hpc::pointer_iterator<array_value_type>, L, std::ptrdiff_t, std::ptrdiff_t>;
  private:
  iterator_type m_iterator;
  public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_reference(iterator_type iterator_in) noexcept : m_iterator(iterator_in) {}
  // movable
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_reference(array_vector_reference&&) = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_reference& operator=(array_vector_reference&&) = default;
  // not copyable
  HPC_HOST_DEVICE array_vector_reference(array_vector_reference const&) = delete;
  HPC_HOST_DEVICE array_vector_reference& operator=(array_vector_reference const&) = delete;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit operator T() const noexcept {
    return ::hpc::array_traits<T>::load(m_iterator);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE void operator=(T const& value) const noexcept {
    ::hpc::array_traits<T>::store(m_iterator, value);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE T load() const noexcept {
    return ::hpc::array_traits<T>::load(m_iterator);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE void store(T const& value) const noexcept {
    ::hpc::array_traits<T>::store(m_iterator, value);
  }
};

template <class T, layout L>
class array_vector_reference<T const, L> {
  public:
  using array_value_type = typename ::hpc::array_traits<T>::value_type;
  using iterator = ::hpc::impl::inner_iterator<::hpc::pointer_iterator<array_value_type const>, L, std::ptrdiff_t, std::ptrdiff_t>;
  private:
  iterator m_iterator;
  public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit array_vector_reference(iterator iterator_in) noexcept : m_iterator(iterator_in) {}
  // movable
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_reference(array_vector_reference&&) = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_reference& operator=(array_vector_reference&&) = default;
  // not copyable
  HPC_HOST_DEVICE array_vector_reference(array_vector_reference const&) = delete;
  HPC_HOST_DEVICE array_vector_reference& operator=(array_vector_reference const&) = delete;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit operator T() const noexcept {
    return ::hpc::array_traits<T>::load(m_iterator);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE T load() const noexcept {
    return ::hpc::array_traits<T>::load(m_iterator);
  }
};

template <class T, layout L, class Index>
class array_vector_iterator {
  public:
  using value_type = std::remove_const_t<T>;
  using array_value_type = typename ::hpc::array_traits<value_type>::value_type;
  using qualified_array_value_type = typename std::conditional<
    std::is_const<T>::value,
    array_value_type const,
    array_value_type>::type;
  using iterator = ::hpc::impl::outer_iterator<::hpc::pointer_iterator<qualified_array_value_type>, L, std::ptrdiff_t, std::ptrdiff_t>;
  private:
  iterator m_iterator;
  public:
  using difference_type = Index;
  using reference = ::hpc::impl::array_vector_reference<T, L>;
  using pointer = T*;
  using iterator_category = typename iterator::iterator_category;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr array_vector_iterator(iterator iterator_in) noexcept : m_iterator(iterator_in) {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(array_vector_iterator const& other) const noexcept {
    return m_iterator == other.m_iterator;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(array_vector_iterator const& other) const noexcept {
    return m_iterator != other.m_iterator;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept {
    auto const inner_range = *m_iterator;
    return reference(inner_range.begin());
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_iterator& operator++() noexcept {
    ++m_iterator;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_iterator;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_iterator& operator--() noexcept {
    --m_iterator;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_iterator;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_iterator& operator+=(difference_type const n) noexcept {
    m_iterator += std::ptrdiff_t(n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array_vector_iterator& operator-=(difference_type const n) noexcept {
    m_iterator -= std::ptrdiff_t(n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr array_vector_iterator operator+(difference_type const n) const noexcept {
    return array_vector_iterator(m_iterator + std::ptrdiff_t(n));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr array_vector_iterator operator-(difference_type const n) const noexcept {
    return array_vector_iterator(m_iterator - std::ptrdiff_t(n));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr difference_type operator-(array_vector_iterator const& other) const noexcept {
    return difference_type(m_iterator - other.m_iterator);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const i) const noexcept {
    return *((*this) + i);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(array_vector_iterator const& other) const noexcept {
    return m_iterator < other.m_iterator;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(array_vector_iterator const& other) const noexcept {
    return m_iterator > other.m_iterator;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(array_vector_iterator const& other) const noexcept {
    return m_iterator <= other.m_iterator;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(array_vector_iterator const& other) const noexcept {
    return m_iterator >= other.m_iterator;
  }
};

}

template <
  class T,
  layout L = ::hpc::host_layout,
  class Allocator = std::allocator<T>,
  class ExecutionPolicy = ::hpc::serial_policy,
  class Index = std::ptrdiff_t>
class array_vector {
  using array_value_type = typename ::hpc::array_traits<T>::value_type;
  static constexpr std::ptrdiff_t array_size() noexcept { return ::hpc::array_traits<T>::size(); }
  using matrix_allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<array_value_type>;
  using matrix_type = ::hpc::matrix<array_value_type, L, matrix_allocator_type, ExecutionPolicy>;
  matrix_type m_matrix;
public:
  using value_type = T;
  using allocator_type = Allocator;
  using execution_policy = ExecutionPolicy;
  using size_type = Index;
  using difference_type = typename matrix_type::difference_type;
  using reference = ::hpc::impl::array_vector_reference<value_type, L>;
  using const_reference = ::hpc::impl::array_vector_reference<value_type const, L>;
  using pointer = T*;
  using const_pointer = T const*;
  using iterator = ::hpc::impl::array_vector_iterator<T, L, Index>;
  using const_iterator = ::hpc::impl::array_vector_iterator<T const, L, Index>;
  constexpr array_vector() noexcept = default;
  array_vector(size_type count)
    :m_matrix(count, array_size())
  {}
  array_vector(allocator_type const& allocator_in, execution_policy const& exec_in) noexcept
    :m_matrix(allocator_in, exec_in)
  {
  }
  array_vector(size_type count, allocator_type const& allocator_in, execution_policy const& exec_in)
    :m_matrix(count, array_size(), allocator_in, exec_in)
  {
  }
  array_vector(array_vector&&) noexcept = default;
  array_vector(array_vector const&) = delete;
  array_vector& operator=(array_vector&&) = default;
  array_vector& operator=(array_vector const&) = delete;
  iterator begin() noexcept { return iterator(m_matrix.begin()); }
  const_iterator begin() const noexcept { return const_iterator(m_matrix.begin()); }
  const_iterator cbegin() const noexcept { return const_iterator(m_matrix.begin()); }
  iterator end() noexcept { return iterator(m_matrix.end()); }
  const_iterator end() const noexcept { return const_iterator(m_matrix.end()); }
  const_iterator cend() const noexcept { return const_iterator(m_matrix.end()); }
  bool empty() const noexcept { return m_matrix.empty(); }
  size_type size() const noexcept { return size_type(m_matrix.size()); }
  void clear() { m_matrix.clear(); }
  void resize(size_type count) { m_matrix.resize(count, array_size()); }
  constexpr allocator_type get_allocator() const noexcept { return m_matrix.get_allocator(); }
  constexpr execution_policy get_execution_policy() const noexcept { return m_matrix.get_execution_policy(); }
  constexpr reference operator[](size_type i) noexcept { return begin()[i]; }
  constexpr const_reference operator[](size_type i) const noexcept { return begin()[i]; }
};

template <class T, class Index = std::ptrdiff_t>
using host_array_vector = array_vector<T, ::hpc::host_layout, ::hpc::host_allocator<T>, ::hpc::host_policy, Index>;
template <class T, class Index = std::ptrdiff_t>
using device_array_vector = array_vector<T, ::hpc::device_layout, ::hpc::device_allocator<T>, ::hpc::device_policy, Index>;
template <class T, class Index = std::ptrdiff_t>
using pinned_array_vector = array_vector<T, ::hpc::device_layout, ::hpc::pinned_allocator<T>, ::hpc::host_policy, Index>;

}
