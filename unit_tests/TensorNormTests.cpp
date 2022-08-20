#include "PlatoTestHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Tet4.hpp"
#include "Mechanics.hpp"
#include "TensorPNorm.hpp"

#include "elliptic/EvaluationTypes.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace TensorNormTests
{

/******************************************************************************/
/*!
  \brief test von Mises stress p-norm with volume scaling as default
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(TensorNormTests, VonMisesPNormDefaultVolumeScaling)
{
    using PhysicsType = Plato::Mechanics<Plato::Tet4>;
    using ElementType = typename PhysicsType::ElementType;
    using ResidualEvalT = Plato::Elliptic::Evaluation<ElementType>::Residual;

    constexpr int numVoigt = ElementType::mNumVoigtTerms;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // set parameters
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='von Mises P-Norm'>                                    \n"
      "  <Parameter name='Type' type='string' value='Scalar Function'/>              \n"
      "  <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm'/>              \n"
      "  <Parameter name='Exponent' type='double' value='6.0'/>              \n"
      "  <ParameterList name='Normalize'>                                    \n"
      "    <Parameter name='Type' type='string' value='Von Mises'/>              \n"
      "  </ParameterList>                                                        \n"
      "</ParameterList>                                                        \n"
    );

    // construct tensor norm
    Plato::TensorNormFactory<numVoigt, ResidualEvalT> tNormFactory;
    auto tNorm = tNormFactory.create(*tParamList);

    // generate synthetic input
    int tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tResult("result workset", tNumCells);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStress("stress", tNumCells, numVoigt);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControl("control workset", tNumCells, tNumNodesPerCell);
    Plato::ScalarVectorT<Plato::Scalar> tCellVolume("cell volume", tNumCells);

    Kokkos::deep_copy(tResult, 0.0);
    Kokkos::deep_copy(tControl, 1.0);
    Kokkos::deep_copy(tCellVolume, 0.5);    

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for (int tVoigtIndex=0; tVoigtIndex<numVoigt; tVoigtIndex++)
        {
            tStress(aCellOrdinal,tVoigtIndex) = 0.1*(tVoigtIndex + 1);
        }
    }, "stress");

    // compute tensor norm
    tNorm->evaluate(tResult, tStress, tControl, tCellVolume);

    auto tResult_Host = Kokkos::create_mirror_view( tResult );
    Kokkos::deep_copy( tResult_Host, tResult );

    TEST_FLOATING_EQUALITY(tResult_Host(0), 6.406452, 1e-13);
}

/******************************************************************************/
/*!
  \brief test von Mises stress p-norm with volume scaling specified
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(TensorNormTests, VonMisesPNormSpecifiedVolumeScaling)
{
    using PhysicsType = Plato::Mechanics<Plato::Tet4>;
    using ElementType = typename PhysicsType::ElementType;
    using ResidualEvalT = Plato::Elliptic::Evaluation<ElementType>::Residual;

    constexpr int numVoigt = ElementType::mNumVoigtTerms;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // set parameters
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='von Mises P-Norm'>                                    \n"
      "  <Parameter name='Type' type='string' value='Scalar Function'/>              \n"
      "  <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm'/>              \n"
      "  <Parameter name='Exponent' type='double' value='6.0'/>              \n"
      "  <ParameterList name='Normalize'>                                    \n"
      "    <Parameter name='Type' type='string' value='Von Mises'/>              \n"
      "    <Parameter name='Volume Scaling' type='bool' value='true'/>              \n"
      "  </ParameterList>                                                        \n"
      "</ParameterList>                                                        \n"
    );

    // construct tensor norm
    Plato::TensorNormFactory<numVoigt, ResidualEvalT> tNormFactory;
    auto tNorm = tNormFactory.create(*tParamList);

    // generate synthetic input
    int tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tResult("result workset", tNumCells);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStress("stress", tNumCells, numVoigt);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControl("control workset", tNumCells, tNumNodesPerCell);
    Plato::ScalarVectorT<Plato::Scalar> tCellVolume("cell volume", tNumCells);

    Kokkos::deep_copy(tResult, 0.0);
    Kokkos::deep_copy(tControl, 1.0);
    Kokkos::deep_copy(tCellVolume, 2.7);    

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for (int tVoigtIndex=0; tVoigtIndex<numVoigt; tVoigtIndex++)
        {
            tStress(aCellOrdinal,tVoigtIndex) = 0.1*(tVoigtIndex + 1);
        }
    }, "stress");

    // compute tensor norm
    tNorm->evaluate(tResult, tStress, tControl, tCellVolume);

    auto tResult_Host = Kokkos::create_mirror_view( tResult );
    Kokkos::deep_copy( tResult_Host, tResult );

    TEST_FLOATING_EQUALITY(tResult_Host(0), 34.5948408, 1e-13);
}

/******************************************************************************/
/*!
  \brief test von Mises stress p-norm with no volume scaling
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(TensorNormTests, VonMisesPNormNoVolumeScaling)
{
    using PhysicsType = Plato::Mechanics<Plato::Tet4>;
    using ElementType = typename PhysicsType::ElementType;
    using ResidualEvalT = Plato::Elliptic::Evaluation<ElementType>::Residual;

    constexpr int numVoigt = ElementType::mNumVoigtTerms;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;

    // set parameters
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='von Mises P-Norm'>                                    \n"
      "  <Parameter name='Type' type='string' value='Scalar Function'/>              \n"
      "  <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm'/>              \n"
      "  <Parameter name='Exponent' type='double' value='6.0'/>              \n"
      "  <ParameterList name='Normalize'>                                    \n"
      "    <Parameter name='Type' type='string' value='Von Mises'/>              \n"
      "    <Parameter name='Volume Scaling' type='bool' value='false'/>              \n"
      "  </ParameterList>                                                        \n"
      "</ParameterList>                                                        \n"
    );

    // construct tensor norm
    Plato::TensorNormFactory<numVoigt, ResidualEvalT> tNormFactory;
    auto tNorm = tNormFactory.create(*tParamList);

    // generate synthetic input
    int tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tResult("result workset", tNumCells);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStress("stress", tNumCells, numVoigt);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControl("control workset", tNumCells, tNumNodesPerCell);
    Plato::ScalarVectorT<Plato::Scalar> tCellVolume("cell volume", tNumCells);

    Kokkos::deep_copy(tResult, 0.0);
    Kokkos::deep_copy(tControl, 1.0);
    Kokkos::deep_copy(tCellVolume, 15.3);    

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
    {
        for (int tVoigtIndex=0; tVoigtIndex<numVoigt; tVoigtIndex++)
        {
            tStress(aCellOrdinal,tVoigtIndex) = 0.1*(tVoigtIndex + 1);
        }
    }, "stress");

    // compute tensor norm
    tNorm->evaluate(tResult, tStress, tControl, tCellVolume);

    auto tResult_Host = Kokkos::create_mirror_view( tResult );
    Kokkos::deep_copy( tResult_Host, tResult );

    TEST_FLOATING_EQUALITY(tResult_Host(0), 12.812904, 1e-13);
}

}
