#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoStaticsTypes.hpp"

#include "ImplicitFunctors.hpp"

#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <Sacado.hpp>

#include "alg/CrsLinearProblem.hpp"
#include "alg/ParallelComm.hpp"

#include "Simp.hpp"
#include "Solutions.hpp"
#include "ScalarProduct.hpp"
#include "WorksetBase.hpp"
#include "elliptic/VectorFunction.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "LinearStress.hpp"
#include "GradientMatrix.hpp"
//#include "geometric/GeometryScalarFunction.hpp"
#include "ApplyConstraints.hpp"
#include "elliptic/Problem.hpp"

#include "Tet10.hpp"
#include "MechanicsElement.hpp"
#include "Mechanics.hpp"
//#include "Thermal.hpp"

#include "SmallStrain.hpp"
#include "GeneralStressDivergence.hpp"

#include <fenv.h>

using ordType = typename Plato::ScalarMultiVector::size_type;

/******************************************************************************/
/*! 
  \brief Load a unit mesh and workset the configuration
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, ConfigWorkset )
{ 
    constexpr int meshWidth=1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    Plato::WorksetBase<ElementType> worksetBase(tMesh);

    auto tNumCells     = tMesh->NumElements();
    auto tNodesPerCell = ElementType::mNumNodesPerCell;
    auto tSpaceDims    = ElementType::mNumSpatialDims;

    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDims);

    worksetBase.worksetConfig(tConfigWS);

    auto tConfigWSHost = Kokkos::create_mirror_view( tConfigWS );
    Kokkos::deep_copy( tConfigWSHost, tConfigWS );

    std::vector<std::vector<std::vector<Plato::Scalar>>> tConfigWSGold =
    {
     {{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {0.5, 0.5, 0.0},
      {0.5, 1.0, 0.0}, {0.0, 0.5, 0.0}, {0.5, 0.5, 0.5}, {1.0, 1.0, 0.5}, {0.5, 1.0, 0.5}},
     {{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 0.5, 0.0},
      {0.0, 1.0, 0.5}, {0.0, 0.5, 0.5}, {0.5, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 1.0, 1.0}},
     {{0.0, 0.0, 0.0}, {0.0, 1.0, 1.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 0.5, 0.5},
      {0.0, 0.5, 1.0}, {0.0, 0.0, 0.5}, {0.5, 0.5, 0.5}, {0.5, 1.0, 1.0}, {0.5, 0.5, 1.0}},
     {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.5},
      {0.5, 0.0, 1.0}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.5}, {0.5, 0.5, 1.0}, {1.0, 0.5, 1.0}},
     {{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {0.5, 0.0, 0.5},
      {1.0, 0.0, 0.5}, {0.5, 0.0, 0.0}, {0.5, 0.5, 0.5}, {1.0, 0.5, 1.0}, {1.0, 0.5, 0.5}},
     {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {0.5, 0.0, 0.0},
      {1.0, 0.5, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.5, 0.5}, {1.0, 0.5, 0.5}, {1.0, 1.0, 0.5}}
    };
  
    int tNumGold_I=tConfigWSGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tConfigWSGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            int tNumGold_K=tConfigWSGold[0][0].size();
            for(int k=0; k<tNumGold_K; k++)
            {
                TEST_FLOATING_EQUALITY(tConfigWSHost(i,j,k), tConfigWSGold[i][j][k], 1e-13);
            }
        }
    }
}

/******************************************************************************/
/*! 
  \brief Load a unit mesh and workset the state
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, StateWorkset )
{ 
    constexpr int meshWidth=1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    Plato::WorksetBase<ElementType> worksetBase(tMesh);

    auto tNumCells     = tMesh->NumElements();
    auto tNumNodes     = tMesh->NumNodes();
    auto tDofsPerCell  = ElementType::mNumDofsPerCell;
    auto tSpaceDims    = ElementType::mNumSpatialDims;

    // create displacement field, u(x) = 0.001*x;
    auto tCoords = tMesh->Coordinates();
    Plato::ScalarVector tDisp("displacement", tCoords.size());
    Kokkos::parallel_for("set displacement", Kokkos::RangePolicy<int>(0, tNumNodes),
    KOKKOS_LAMBDA(int nodeOrdinal)
    {
      tDisp(tSpaceDims*nodeOrdinal) = 0.001*tCoords(tSpaceDims*nodeOrdinal);
    });

    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);

    worksetBase.worksetState(tDisp, tStateWS);

    auto tStateWSHost = Kokkos::create_mirror_view( tStateWS );
    Kokkos::deep_copy( tStateWSHost, tStateWS );

    Plato::Scalar tHf = 0.0005, tHl = 0.001;

    std::vector<std::vector<std::vector<Plato::Scalar>>> tStateWSGold =
    {
     {{0.0, 0.0, 0.0}, {tHl, 0.0, 0.0}, {0.0, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHf, 0.0, 0.0},
      {tHf, 0.0, 0.0}, {0.0, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHf, 0.0, 0.0}},
     {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {tHl, 0.0, 0.0}, {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}},
     {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {tHl, 0.0, 0.0}, {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}},
     {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHl, 0.0, 0.0}, {0.0, 0.0, 0.0},
      {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHl, 0.0, 0.0}},
     {{0.0, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHf, 0.0, 0.0},
      {tHl, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHl, 0.0, 0.0}},
     {{0.0, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHf, 0.0, 0.0},
      {tHl, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHf, 0.0, 0.0}, {tHl, 0.0, 0.0}, {tHl, 0.0, 0.0}}
    };
  
    int tNumGold_I=tStateWSGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tStateWSGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            int tNumGold_K=tStateWSGold[0][0].size();
            for(int k=0; k<tNumGold_K; k++)
            {
                TEST_FLOATING_EQUALITY(tStateWSHost(i,3*j+k), tStateWSGold[i][j][k], 1e-13);
            }
        }
    }
}

/******************************************************************************/
/*! 
  \brief Load a unit mesh and compute the gradient matrix
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, ComputeGradientMatrix )
{ 
  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

  Plato::WorksetBase<ElementType> worksetBase(tMesh);

  auto tNumCells     = tMesh->NumElements();
  auto tNodesPerCell = ElementType::mNumNodesPerCell;
  auto tSpaceDims    = ElementType::mNumSpatialDims;

  Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDims);

  worksetBase.worksetConfig(tConfigWS);

  Plato::ComputeGradientMatrix<ElementType> computeGradientMatrix;

  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();

  Kokkos::View<Plato::Scalar****, Plato::Layout, Plato::MemSpace>
  tGradientsView("all gradients", tNumCells, tNumPoints, tNodesPerCell, tSpaceDims);

  Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
  {
      Plato::Scalar tVolume(0.0);

      Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, Plato::Scalar> tGradient;
      auto tCubPoint = tCubPoints(gpOrdinal);
      computeGradientMatrix(cellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);
      tVolume *= tCubWeights(gpOrdinal);

      for(int I=0; I<ElementType::mNumNodesPerCell; I++)
        for(int i=0; i<ElementType::mNumSpatialDims; i++)
          tGradientsView(cellOrdinal, gpOrdinal, I, i) = tGradient(I,i);
  });

  auto tGradientsHost = Kokkos::create_mirror_view( tGradientsView );
  Kokkos::deep_copy( tGradientsHost, tGradientsView );

  std::vector<std::vector<Plato::Scalar>> tGradientsGold = {
     { 0.000000000000000000,  0.447213595499963701,  0.000000000000000000},
     { 1.341640786499876060,  0.000000000000000000, -1.341640786499876730},
     { 0.447213595499955985, -0.447213595499955985,  0.000000000000000000},
     { 0.000000000000000000,  0.000000000000000000, -0.447213595499955985},
     { 0.552786404500035911, -2.341640786499875840, -0.552786404500035022},
     {-1.788854381999831710,  2.341640786499875840, -0.552786404500044348},
     {-0.552786404500036466,  0.000000000000000000,  0.000000000000000000},
     {-0.000000000000000000, -0.552786404500044015,  0.552786404500036688},
     { 0.552786404500044126,  0.000000000000000000,  1.788854381999831490},
     {-0.552786404500044015,  0.552786404500044015,  0.552786404500044015}
  };

  int tNumGold_J=tGradientsGold.size();
  for(int j=0; j<tNumGold_J; j++)
  {
      int tNumGold_K=tGradientsGold[j].size();
      for(int k=0; k<tNumGold_K; k++)
      {
          if(tGradientsGold[j][k] == 0)
          {
              TEST_ASSERT(fabs(tGradientsHost(0,0,j,k)) < 1e-14);
          }
          else
          {
              TEST_FLOATING_EQUALITY(tGradientsHost(0,0,j,k), tGradientsGold[j][k], 1e-13);
          }
      }
  }
}

/******************************************************************************/
/*! 
  \brief Load a unit mesh and compute the cell stresses
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, ComputeStresses )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>      \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  auto tOnlyDomain = tSpatialModel.Domains.front();

  using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

  Plato::WorksetBase<ElementType> worksetBase(tMesh);

  auto tNumNodes = tMesh->NumNodes();
  auto tNumCells = tMesh->NumElements();

  constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
  constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
  constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
  constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

  // create displacement field, u(x) = 0.001*x;
  auto tCoords = tMesh->Coordinates();
  Plato::ScalarVector tDisp("displacement", tCoords.size());
  Kokkos::parallel_for("set displacement", Kokkos::RangePolicy<int>(0, tNumNodes),
  KOKKOS_LAMBDA(int nodeOrdinal)
  {
    tDisp(tSpaceDims*nodeOrdinal) = 0.001*tCoords(tSpaceDims*nodeOrdinal);
  });

  Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDims);
  worksetBase.worksetConfig(tConfigWS);

  Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
  worksetBase.worksetState(tDisp, tStateWS);

  Plato::ComputeGradientMatrix<ElementType> computeGradient;
  Plato::SmallStrain<ElementType> voigtStrain;

  Plato::ElasticModelFactory<tSpaceDims> mmfactory(*tParamList);
  auto materialModel = mmfactory.create(tOnlyDomain.getMaterialName());
  auto tCellStiffness = materialModel->getStiffnessMatrix();

  Plato::LinearStress<Plato::Elliptic::ResidualTypes<ElementType>, ElementType> voigtStress(tCellStiffness);

  Plato::GeneralStressDivergence<ElementType>  stressDivergence;

  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();

  Plato::ScalarMultiVectorT<Plato::Scalar> tCellStress("stress", tNumCells, tNumVoigtTerms);
  Plato::ScalarMultiVectorT<Plato::Scalar> tCellStrain("strain", tNumCells, tNumVoigtTerms);
  Plato::ScalarVectorT<Plato::Scalar> tCellVolume("cell volume", tNumCells);

  Plato::ScalarMultiVectorT<Plato::Scalar> tResult("result", tNumCells, tDofsPerCell);

  Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
  {
      Plato::Scalar tVolume(0.0);

      Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, Plato::Scalar> tGradient;

      Plato::Array<ElementType::mNumVoigtTerms, Plato::Scalar> tStrain(0.0);
      Plato::Array<ElementType::mNumVoigtTerms, Plato::Scalar> tStress(0.0);

      auto tCubPoint = tCubPoints(gpOrdinal);

      computeGradient(cellOrdinal, tCubPoint, tConfigWS, tGradient, tVolume);

      voigtStrain(cellOrdinal, tStrain, tStateWS, tGradient);

      voigtStress(tStress, tStrain);

      tVolume *= tCubWeights(gpOrdinal);
      tCellVolume(cellOrdinal) += tVolume;

      stressDivergence(cellOrdinal, tResult, tStress, tGradient, tVolume);

      for(int i=0; i<ElementType::mNumVoigtTerms; i++)
      {
          tCellStrain(cellOrdinal,i) += tVolume*tStrain(i);
          tCellStress(cellOrdinal,i) += tVolume*tStress(i);
      }
  });

  Kokkos::parallel_for("average", Kokkos::RangePolicy<int>(0, tNumCells),
  KOKKOS_LAMBDA(int cellOrdinal)
  {
      for(int i=0; i<ElementType::mNumVoigtTerms; i++)
      {
          tCellStress(cellOrdinal,i) /= tCellVolume(cellOrdinal);
          tCellStrain(cellOrdinal,i) /= tCellVolume(cellOrdinal);
      }
  });

  auto tCellStrainHost = Kokkos::create_mirror_view( tCellStrain );
  Kokkos::deep_copy( tCellStrainHost, tCellStrain );

  std::vector<std::vector<Plato::Scalar>> tCellStrainGold = {
    {0.001, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0, 0.0, 0.0, 0.0}
  };

  int tNumGold_I=tCellStrainGold.size();
  for(int iCell=0; iCell<tNumGold_I; iCell++)
  {
      int tNumGold_J=tCellStrainGold[0].size();
      for(int j=0; j<tNumGold_J; j++)
      {
          if(tCellStrainGold[iCell][j] == 0.0)
          {
              TEST_ASSERT(fabs(tCellStrainHost(iCell,j)) < 1e-15);
          }
          else
          {
              TEST_FLOATING_EQUALITY(tCellStrainHost(iCell,j), tCellStrainGold[iCell][j], 1e-13);
          }
      }
  }

  auto tCellStressHost = Kokkos::create_mirror_view( tCellStress );
  Kokkos::deep_copy( tCellStressHost, tCellStress );

  std::vector<std::vector<Plato::Scalar>> tCellStressGold = {
    {1346.15384615384573, 576.923076923076792, 576.923076923076792, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
    {1346.15384615384573, 576.923076923076792, 576.923076923076792, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
    {1346.15384615384573, 576.923076923076792, 576.923076923076792, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
    {1346.15384615384573, 576.923076923076792, 576.923076923076792, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
    {1346.15384615384573, 576.923076923076792, 576.923076923076792, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000},
    {1346.15384615384573, 576.923076923076792, 576.923076923076792, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000}
  };

  tNumGold_I=tCellStressGold.size();
  for(int iCell=0; iCell<tNumGold_I; iCell++)
  {
      int tNumGold_J=tCellStressGold[0].size();
      for(int j=0; j<tNumGold_J; j++)
      {
          if(tCellStressGold[iCell][j] == 0.0)
          {
              TEST_ASSERT(fabs(tCellStressHost(iCell,j)) < 1e-10);
          }
          else
          {
              TEST_FLOATING_EQUALITY(tCellStressHost(iCell,j), tCellStressGold[iCell][j], 1e-10);
          }
      }
  }

  auto tResultHost = Kokkos::create_mirror_view( tResult );
  Kokkos::deep_copy( tResultHost, tResult );

  std::vector<std::vector<Plato::Scalar>> tResultGold =
{{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
  224.3589743589730, -96.15384615384630, -96.15384615384559, 0., 
  96.15384615384630, -96.15384615384630, -224.3589743589730, 0., 0., 
  0., -96.15384615384633, 96.15384615384559, 224.3589743589747, 0., 
  0., -224.3589743589747, 96.15384615384633, 96.15384615384633}, {0., 
  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
  0., -96.15384615384556, -224.3589743589746, 96.15384615384630, 
  0., -224.3589743589729, -96.15384615384630, 96.15384615384556, 
  224.3589743589729, -96.15384615384630, 0., 224.3589743589747, 
  96.15384615384630, -96.15384615384630, 0., 0., 
  96.15384615384630}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
  0., -224.3589743589729, 
  96.15384615384556, -96.15384615384630, -224.3589743589746, 0., 
  96.15384615384630, 0., -96.15384615384556, 0., 224.3589743589729, 
  0., -96.15384615384630, 0., 96.15384615384630, 0., 
  224.3589743589747, -96.15384615384630, 96.15384615384630}, {0., 0., 
  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -224.3589743589730, 0., 0., 
  0., -96.15384615384630, 96.15384615384630, 
  224.3589743589730, -96.15384615384559, -96.15384615384630, 0., 
  96.15384615384558, -96.15384615384630, -224.3589743589747, 
  96.15384615384633, 96.15384615384630, 224.3589743589747, 0., 
  0.}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
  0., -224.3589743589747, -96.15384615384554, 96.15384615384556, 
  224.3589743589747, -96.15384615384632, 0., 0., 
  0., -96.15384615384556, -224.3589743589747, 96.15384615384556, 0., 
  0., 0., 96.15384615384633, 224.3589743589747, 
  96.15384615384630, -96.15384615384633}, {0., 0., 0., 0., 0., 0., 0.,
   0., 0., 0., 0., 0., 0., -96.15384615384556, 0., 224.3589743589747, 
  0., -96.15384615384630, -224.3589743589747, 
  96.15384615384556, -96.15384615384555, -224.3589743589747, 0., 
  96.15384615384556, 224.3589743589747, -96.15384615384630, 
  96.15384615384630, 0., 96.15384615384633, 0.}};

  tNumGold_I=tResultGold.size();
  for(int iCell=0; iCell<tNumGold_I; iCell++)
  {
      int tNumGold_J=tResultGold[0].size();
      for(int j=0; j<tNumGold_J; j++)
      {
          if(tResultGold[iCell][j] == 0.0)
          {
              TEST_ASSERT(fabs(tResultHost(iCell,j)) < 1e-11);
          }
          else
          {
              TEST_FLOATING_EQUALITY(tResultHost(iCell,j), tResultGold[iCell][j], 1e-11);
          }
      }
  }
}

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, ElastostaticResidual3D )
{
  // create test mesh
  //
  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

  // create mesh based density
  //
  auto tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector z("density", tNumNodes);
  Kokkos::deep_copy(z, 1.0);


  // create displacement field, u(x) = 0.001*x;
  //
  auto tCoords = tMesh->Coordinates();
  auto tSpaceDims = tMesh->NumDimensions();
  Plato::ScalarVector u("displacement", tCoords.size());
  Kokkos::parallel_for("set displacement", Kokkos::RangePolicy<int>(0, tNumNodes),
  KOKKOS_LAMBDA(int nodeOrdinal)
  {
    u(tSpaceDims*nodeOrdinal) = 0.001*tCoords(tSpaceDims*nodeOrdinal);
  });


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
    "  <ParameterList name='Elliptic'>                                                \n"
    "    <ParameterList name='Penalty Function'>                                      \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Criteria'>                                                \n"
    "    <ParameterList name='Internal Elastic Energy'>                               \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>             \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  using PhysicsType = typename Plato::Mechanics<Plato::Tet10>;

  Plato::Elliptic::VectorFunction<PhysicsType>
    tVectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto residual = tVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold =
    {0., 0., 0., -224.358974358972972, -96.1538461538455493, 0., 0., 0.,
    0., -224.358974358973001, 0., -96.1538461538455493,
    -448.717948717945887, 0., 0., -224.358974358974649, 0.,
    96.1538461538463025, 0., 0., 0., -224.358974358974649,
    96.1538461538463025, 0., 0., 0., 0., 0., -96.1538461538455493,
    -96.1538461538455493, 0., -192.307692307691127, 0., 0.,
    -96.1538461538463025, 96.1538461538463025, 0., 0.,
    -192.307692307691127, 0., 0., 0., 0., 0., 192.307692307692605, 0.,
    96.1538461538463025, -96.1538461538463025, 0., 192.307692307692633,
    0., 0., 96.1538461538463025, 96.1538461538463025, 0., 0., 0.,
    224.358974358974677, -96.1538461538463167, 0., 0., 0., 0.,
    224.358974358974677, 0., -96.1538461538463025, 448.717948717949355,
    0., 0., 224.358974358974649, 0., 96.1538461538463309, 0., 0., 0.,
    224.358974358974706, 96.1538461538463309, 0., 0., 0., 0.};

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-11);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test constraint gradient wrt state, u. (i.e., jacobian)
  //
  auto jacobian = tVectorFunction.gradient_u(u,z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = 
{423076.923076922947, 0., 0., 0., 423076.923076922889, 0., 0., 0.,
423076.923076922889, -102564.102564102563, 0., 38461.5384615380899,
0., -102564.102564102563, 38461.5384615380899, 57692.3076923071276,
57692.3076923071276, -358974.358974358824, 25641.0256410256552, 0.,
-12820.5128205128185, 0., 25641.0256410256552, -12820.5128205128185,
-19230.7692307692269, -19230.7692307692269, 89743.5897435897350,
-102564.102564102563, 38461.5384615380899, 0., 57692.3076923071276,
-358974.358974358824, 57692.3076923071276, 0., 38461.5384615380899,
-102564.102564102563};

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    if(gold_jac_entries[i] == 0.0){
      TEST_ASSERT(fabs(jac_entriesHost(i)) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
    }
  }


  // compute and test gradient wrt control, z
  //
  auto gradient_z = tVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
-30.1009150817276634, -12.9003921778832886, -12.9003921778832868,
-3.83250882291512918, -1.64250378124933993, 17.2005229038443836,
5.01681918028794716, 2.15006536298054840, -4.30013072596109680,
-3.83250882291512829, 17.2005229038443836, -1.64250378124933971,
-7.66501764583024325, 8.60026145192219360, 8.60026145192219360,
16.2347678982366581, 0, -6.95775767067285500, 5.01681918028794627,
-4.30013072596109680, 2.15006536298054840, 16.2347678982366581,
-6.95775767067285500, 0, 10.0336383605758943, -2.15006536298054840,
-2.15006536298054840, 40.1345534423035701, -1.64250378124933838,
-1.64250378124933927
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    if(gold_grad_entries[i] == 0.0){
      TEST_ASSERT(fabs(grad_entriesHost(i)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
    }
  }

#ifdef NOPE

  // compute and test gradient wrt node position, x
  //
  auto gradient_x = tVectorFunction.gradient_x(u,z);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
-1903.84615384615336, -1903.84615384615336, -1903.84615384615358,
-634.615384615384301, -634.615384615384414, -634.615384615384642,
-211.538461538461490, -211.538461538461462, -211.538461538461661,
-105.769230769230603, 9.61538461538454214, 451.923076923076962,
-163.461538461538652, -48.0769230769230802, -124.999999999999716,
961.538461538461206, 730.769230769230603, 365.384615384615358,
-144.230769230769113, 374.999999999999716, 9.61538461538462741,
942.307692307692150, 596.153846153846189, 634.615384615384301,
-221.153846153846104, -394.230769230769113, -67.3076923076922782,
-942.307692307692150, -307.692307692307395, -230.769230769230688,
548.076923076922867, 317.307692307692150, 278.846153846153811,
663.461538461538112, 259.615384615384528, 221.153846153846132
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

#endif

}

/******************************************************************************/
/*! 
  \brief Test natural BCs in ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, ElastostaticResidual3D_NaturalBC )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

  // create mesh based density
  //
  auto tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector z("density", tNumNodes);
  Kokkos::deep_copy(z, 1.0);


  // create displacement field, u(x) = 0.0;
  //
  auto tCoords = tMesh->Coordinates();
  auto tSpaceDims = tMesh->NumDimensions();
  Plato::ScalarVector u("displacement", tCoords.size());


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
    "  <ParameterList name='Elliptic'>                                                \n"
    "    <ParameterList name='Penalty Function'>                                      \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                            \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>                   \n"
    "      <Parameter  name='Type'     type='string'        value='Uniform'/>         \n"
    "      <Parameter  name='Values'   type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
    "      <Parameter  name='Sides'    type='string'        value='x+'/>              \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  using PhysicsType = typename Plato::Mechanics<Plato::Tet10>;

  Plato::Elliptic::VectorFunction<PhysicsType>
    tVectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto residual = tVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.0416667, 0.,
    0., 0., 0., 0., -0.0416667, 0., 0., 0., 0., 0., -0.0416667, 0., 0., -0.0833333,
    0., 0., -0.0833333, 0., 0., -0.0833333, 0., 0., -0.0416667, 0., 0., 0., 0., 0., 
    -0.0833333, 0., 0., 0., 0., 0., -0.0833333, 0., 0., 0., 0., 0., -0.0416667, 0.,
    0., -0.0833333, 0., 0., -0.0833333, 0., 0., -0.0833333, 0., 0., -0.0416667, 0.,
    0., 0., 0., 0., -0.0416667, 0., 0., 0., 0., 0., -0.0416667, 0., 0., 0., 0., 0.
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-11);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-6);
    }
  }
} 

/******************************************************************************/
/*! 
  \brief Test natural BCs in ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, ElastostaticResidual3D_Solution )
{
    // create test mesh
    //
    constexpr int meshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

    // create input
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                             \n"
      "  <ParameterList name='Spatial Model'>                                           \n"
      "    <ParameterList name='Domains'>                                               \n"
      "      <ParameterList name='Design Volume'>                                       \n"
      "        <Parameter name='Element Block' type='string' value='body'/>             \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                   \n"
      "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
      "  <ParameterList name='Elliptic'>                                                \n"
      "    <ParameterList name='Penalty Function'>                                      \n"
      "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
      "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList name='Material Models'>                                         \n"
      "    <ParameterList name='Unobtainium'>                                           \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
      "      </ParameterList>                                                           \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                            \n"
      "    <ParameterList  name='Traction Vector Boundary Condition'>                   \n"
      "      <Parameter  name='Type'     type='string'        value='Uniform'/>         \n"
      "      <Parameter  name='Values'   type='Array(double)' value='{1.0, 0.0, 0.0}'/> \n"
      "      <Parameter  name='Sides'    type='string'        value='x+'/>              \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "  <ParameterList  name='Essential Boundary Conditions'>                          \n"
      "    <ParameterList  name='X Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='0'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='1'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>              \n"
      "      <Parameter  name='Type'     type='string' value='Zero Value'/>             \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                      \n"
      "      <Parameter  name='Sides'    type='string' value='x-'/>                     \n"
      "    </ParameterList>                                                             \n"
      "  </ParameterList>                                                               \n"
      "</ParameterList>                                                                 \n"
    );


    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    Plato::Elliptic::Problem<Plato::Mechanics<Plato::Tet10>>
        tElasticityProblem(tMesh, *tParamList, tMachine);


    // SOLVE ELASTOSTATICS EQUATIONS
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    auto tElasticitySolution = tElasticityProblem.solution(tControl);

    // TEST RESULTS    
    const Plato::OrdinalType tTimeStep = 0;
    auto tState = tElasticitySolution.get("State");
    auto tSolution = Kokkos::subview(tState, tTimeStep, Kokkos::ALL());
    auto tHostSolution = Kokkos::create_mirror_view(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    std::vector<Plato::Scalar> tGold = 
{8.44215e-8, 9.58193e-7, -7.30424e-8, 4.50125e-9, 
 9.61752e-7, -7.46016e-8, -7.46016e-8, 
 9.68308e-7, -7.43541e-8, -1.50715e-7, 
 9.67836e-7, -1.47979e-7, 1.60339e-7, 
 9.65735e-7, -1.47873e-7, 8.41994e-8, 
 9.6498e-7, -1.49664e-7, 4.12353e-9, 
 9.68308e-7, -1.50715e-7, -7.43541e-8, 
 9.79216e-7, -1.52588e-7, -1.52588e-7};


    Plato::OrdinalType tDofOffset = 350; // comparing only the last 25 dofs
    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tDofIndex=0; tDofIndex < tGold.size(); tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostSolution(tDofOffset+tDofIndex), tGold[tDofIndex], tTolerance);
    }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, InternalElasticEnergy3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

  // create mesh based density from host data
  //
  auto tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector z("density", tNumNodes);
  Kokkos::deep_copy(z, 1.0);

  // create mesh based displacement from host data
  //
  ordType tNumDofs = tMesh->NumDimensions()*tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  std::string tMyFunction("Internal Elastic Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<Plato::Tet10>>
    eeScalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution,z);

  Plato::Scalar value_gold = 1206.13846153846043;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  tSolution.set("State", U);
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = {
   0., 0., 0., -2432.692307692301, -1663.461538461530,
   -615.3846153846263, 0., 0., 0., -2432.692307692299,
   -1663.461538461529, -615.3846153846263, 0., 0., 0.,
   -2355.769230769225, -692.3076923077053, -1432.692307692301,
   -3711.538461538447, -1153.846153846157, -1000.000000000002,
   -3711.538461538460, -1153.846153846169, -999.9999999999920,
   -3711.538461538446, -1153.846153846156, -1000.000000000002,
   -1355.769230769233 };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
      if(grad_u_gold[iNode] == 0.0)
      {
          TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-10);
      }
      else
      {
          TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
      }
  }


  // compute and test criterion gradient wrt control, z
  //
  tSolution.set("State", U);
  auto grad_z = eeScalarFunction.gradient_z(tSolution,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
    -7.53836538461538375, 10.0511538461537988, -10.0511538461538468,
    10.0511538461537935, -2.51278846153846125, 10.0511538461537988,
    10.0511538461537988, 15.0767307692307462, 10.0511538461537935,
    5.02557692307694648, -10.0511538461538485, 15.0767307692307444,
    -15.0767307692307710, 15.0767307692307391, -5.02557692307692072,
    10.0511538461537953, 10.0511538461537917, 15.0767307692307426};

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  tSolution.set("State", U);
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
    0., 0., 0., 91.0903846153847354, -21.9865384615381778,
    5.65384615384538058, 0., 0., 0., 91.0903846153847496,
   -21.9865384615381956, 5.65384615384537881, 0., 0., 0.,
    84.1673076923079861, 26.8846153846146265, -44.8788461538458705,
    75.4500000000003013, 35.1923076923072244, 7.03846153846112799,
    75.4500000000003723, 35.1923076923068976, 7.03846153846193090,
    75.4500000000002871, 35.1923076923071747, 7.03846153846111378
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
      if(grad_x_gold[iNode] == 0.0)
      {
          TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
      }
      else
      {
          TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-12);
      }
  }
}

#ifdef NOPE

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, StressPNorm3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Globalized Stress'>                                  \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm'/>  \n"
    "      <Parameter name='Exponent' type='double' value='12.0'/>                 \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);


  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based displacement from host data
  //
  ordType tNumDofs = spaceDim*tMesh->NumNodes();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Plato::DataMap tDataMap;
  std::string tMyFunction("Globalized Stress");
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 12164.73465517308;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -136045.3530811711, -117804.6353496175, -111724.3961057662,
  -194947.6707559799, -173058.8094781152, -12768.50241208767,
  -58902.31767480877, -55254.17412849799,  98955.89369367871,
  -193123.5989828244, -14592.57418524276, -163938.4506123385,
  -368006.4802340950, -21888.86127786443, -18240.71773155366,
  -174882.8812512706, -7296.287092621476,  145697.7328807848,
  -57078.24590165332,  103212.0611643744, -52214.05450657228,
  -173058.8094781151,  151169.9482002509, -5472.215319466181,
  -115980.5635764621,  47957.88703587653,  46741.83918710628,
  -20064.78950470938, -165762.5223854940, -158466.2352928726,
  -21888.86127786437, -324228.7576783666, -7296.287092621540,
  -1824.071773155300, -158466.2352928726,  151169.9482002512
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-12);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   380.1479579741605, 506.8639439655473, 126.7159859913868,
   506.8639439655473, 760.2959159483208, 253.4319719827737,
   126.7159859913868, 253.4319719827736, 126.7159859913870,
   506.8639439655479, 760.2959159483213, 253.4319719827739,
   760.2959159483212, 1520.591831896641, 760.2959159483211,
   253.4319719827736, 760.2959159483206, 506.8639439655473,
   126.7159859913873, 253.4319719827741, 126.7159859913868,
   253.4319719827742, 760.2959159483217, 506.8639439655475,
   126.7159859913868, 506.8639439655471, 380.1479579741601
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   1889.624352503137,  573.5565681715406,  134.8673067276750,
   1929.468920298000,  558.6789827717420,  228.4649895877097,
   39.84456779486254, -14.87758539979850,  93.59768286003472,
   1880.218982422804,  668.9783228047302,  96.27678827685651,
   1950.502747932197,  734.6449066383240,  244.8816355461080,
   70.28376550939277,  65.66658383359324,  148.6048472692513,
  -9.405370080332322,  95.42175463319003, -38.59051845081814,
   21.03382763419772,  175.9659238665815,  16.41664595839830,
   30.43919771453029,  80.54416923339170,  55.00716440921650,
   1859.185154788609,  493.0123989381497,  79.86014231845859,
   1908.435092663803,  382.7130589051604,  212.0483436293115,
   49.24993787519501, -110.2993400329887,  132.1882013108530,
   1809.935216913413,  603.3117389711377, -52.32805899239442,
   0.000000000000000,  0.000000000000000,  0.0000000000000000
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}

#endif
