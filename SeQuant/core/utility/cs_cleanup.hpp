#ifndef SEQUANT_CORE_UTILITY_CS_CLEANUP_HPP
#define SEQUANT_CORE_UTILITY_CS_CLEANUP_HPP

#include <SeQuant/core/rational.hpp>
#include <SeQuant/core/tensor.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view.hpp>
#include <string>
#include <vector>

namespace sequant {

template <typename TArray>
TArray cleanup_tensor(const TArray& array, const sequant::Tensor& tensor) {
  size_t total_rank = array.trange().rank();

  // Get bra rank and ket rank from the tensor object
  auto bra_rank = tensor.bra_rank();
  auto ket_rank = tensor.ket_rank();

  // Early return for low-rank tensors
  if (total_rank <= 4) {
    return array;
  }

  // Create index vectors for bra and ket
  auto bra_idx = ranges::views::iota(size_t{0}, bra_rank) | ranges::to_vector;
  auto ket_idx = ranges::views::iota(bra_rank, total_rank) | ranges::to_vector;

  // Helper function to convert ordinals to annotation string
  auto ords_to_annot = [](auto const& ords) {
    using ranges::views::intersperse;
    using ranges::views::join;
    using ranges::views::transform;
    auto to_str = [](auto x) { return std::to_string(x); };
    return ords | transform(to_str) | intersperse(std::string{","}) | join |
           ranges::to<std::string>;
  };

  // Create the left annotation (original ordering)
  const auto l_annot =
      ords_to_annot(ranges::views::iota(size_t{0}, total_rank));

  // Initialize cleaned tensor
  TArray cleaned(array.world(), array.trange(), array.shape());
  cleaned.fill(0.0);

  // Initialize permutation sum tensor
  TArray perm_sum(array.world(), array.trange(), array.shape());
  perm_sum.fill(0.0);

  // We need to symmetrize over the ket indices
  // For conventional CC, ket indices are the occupied indices

  // Calculate the symmetrization factor
  rational inv_factor = rational(1, factorial(ket_rank));
  double inv_factor_d = static_cast<double>(inv_factor);

  // Generate all permutations of ket indices
  std::vector<size_t> ket_perm = ket_idx;

  do {
    // Construct the permuted annotation
    // Combine bra indices (unchanged) with permuted ket indices
    std::vector<size_t> full_perm;
    full_perm.reserve(total_rank);

    // Add bra indices (unchanged)
    full_perm.insert(full_perm.end(), bra_idx.begin(), bra_idx.end());

    // Add permuted ket indices
    full_perm.insert(full_perm.end(), ket_perm.begin(), ket_perm.end());

    // Convert to annotation string
    std::string perm_annot = ords_to_annot(full_perm);

    // Accumulate the permuted tensor
    perm_sum(l_annot) += array(perm_annot);

  } while (std::next_permutation(ket_perm.begin(), ket_perm.end()));

  // Apply the symmetrization factor
  perm_sum(l_annot) = perm_sum(l_annot) * inv_factor_d;

  // Compute cleaned = array - perm_sum
  cleaned(l_annot) = array(l_annot) - perm_sum(l_annot);

  // Wait for lazy operations to complete
  TArray::wait_for_lazy_cleanup(cleaned.world());

  return cleaned;
}

}  // namespace sequant

#endif  // SEQUANT_CORE_UTILITY_CS_CLEANUP_HPP
