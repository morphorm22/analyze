/*!
  These unit tests are for the Derivative functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
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
//#include "elliptic/SolutionFunction.hpp"
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
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET10", meshWidth);

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
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET10", meshWidth);

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
    LAMBDA_EXPRESSION(int nodeOrdinal)
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
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET10", meshWidth);

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
  LAMBDA_EXPRESSION(const int cellOrdinal, const int gpOrdinal)
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

  std::vector<std::vector<std::vector<std::vector<Plato::Scalar>>>> tGradientsGold = {
  {
   {
    { 0.30901699437495400,  0.50000000000000555,  0.50000000000000821},
    { 2.42705098312485035, -0.92705098312484934, -0.41458980337503759},
    { 0.49999999999999694, -0.80901699437494489, -0.19098300562505546},
    { 0.00000000000000000,  0.00000000000000000, -0.44721359549995592},
    {-0.61803398874991843, -2.99999999999999556, -2.78885438199984437},
    {-1.61803398874988691,  3.85410196624969004,  0.82917960675008073},
    {-0.99999999999999689,  0.38196601125009521, -0.38196601125010748},
    { 1.00000000000000533, -0.38196601125010887,  2.17082039324993614},
    {-0.61803398874989590,  1.00000000000000488,  0.78885438199983914},
    {-0.38196601125010942, -0.61803398874989601, -0.06524758424986293}
   },
   {
    { 0.30901699437495311,  0.50000000000000632, -0.50000000000000732},
    {-0.80901699437494545,  0.30901699437494828,  1.30901699437494456},
    {-1.50000000000000111,  2.42705098312485124,  1.08541019662497629},
    { 0.00000000000000000,  0.00000000000000000, -0.44721359549995598},
    { 0.61803398874988291, -1.00000000000000044, -0.99999999999998390},
    { 3.61803398874990467, -0.61803398874990223, -6.40688837074974504},
    {-2.23606797749979380, -1.61803398874990334,  3.06524758424985855},
    { 1.00000000000000555, -0.38196601125010920, -1.06524758424986032},
    {-0.61803398874989656,  1.00000000000000555,  2.78885438199983771},
    {-0.38196601125010898, -0.61803398874989634,  1.17082039324993481}
   },
   {
    { 0.32725424859374036, -0.37500000000000433,  0.055901699437494616},
    { 0.04774575140626319,  0.32725424859373475,  0.008155948031232795},
    {-0.37499999999999789,  0.04774575140626313,  0.383155948031229654},
    { 0.00000000000000000,  0.00000000000000000,  1.341640786499875610},
    {-0.46352549156242128,  0.05901699437495373, -0.079179606750063594}, 
    { 0.40450849718747417, -0.46352549156242223, -0.483688103937539637}, 
    { 0.05901699437494125,  0.40450849718747517, -0.542705098312477707}, 
    {-0.25000000000000161, -1.71352549156241918,  0.510081306187553318},
    { 1.96352549156242029, -0.25000000000000127, -1.453444185374861860},
    {-1.71352549156241873,  1.96352549156242051,  0.260081306187556760}
   },
   {
    {-0.92705098312484457, -1.49999999999999223,  2.012461179749799100},
    {-0.80901699437494567,  0.30901699437494839,  0.309016994374945619},
    { 0.49999999999999750, -0.80901699437494589,  0.809016994374943454},
    { 0.00000000000000000,  0.00000000000000000, -0.447213595499955985},
    { 3.85410196624967938, -2.23606797749979958, -0.788854381999823384},
    { 0.38196601125010898,  0.61803398874989667, -1.381966011250109090},
    {-2.99999999999999556,  3.61803398874989313, -3.406888370749711740},
    { 1.00000000000000555, -0.38196601125010932,  0.170820393249938085},
    {-0.61803398874989657,  1.00000000000000600, -0.447213595499959093},
    {-0.38196601125010898, -0.61803398874989667,  3.170820393249933480}
   }
  },
  {
   {
    { 0.80901699437496232,  0.80901699437495944, -0.30901699437495422},
    { 2.01246117974981287,  1.50000000000000067, -2.42705098312485079},
    { 0.30901699437494145, -0.30901699437494806, -0.49999999999999700},
    {-0.44721359549995598,  0.00000000000000000,  0.00000000000000000},
    {-3.40688837074976192, -3.61803398874991400,  0.61803398874991843},
    {-0.78885438199980650,  2.23606797749980313,  1.61803398874988713},
    {-1.38196601125010421, -0.61803398874990156,  0.99999999999999711},
    { 3.17082039324994147,  0.61803398874989634, -1.00000000000000555},
    { 0.17082039324994319,  0.38196601125010898,  0.61803398874989601},
    {-0.44721359549997225, -1.00000000000000533,  0.38196601125010953}
   },
   {
    {-0.19098300562505410,  0.80901699437495922, -0.30901699437495311},
    { 0.49999999999999900, -0.49999999999999705,  0.80901699437494534},
    {-0.41458980337502554,  0.92705098312484990,  1.50000000000000089},
    {-0.44721359549995598,  0.00000000000000000,  0.00000000000000000},
    {-0.38196601125010065, -0.38196601125011747, -0.61803398874988280},
    {-2.78885438199983815,  3.00000000000000222, -3.61803398874990378},
    { 0.82917960675006385, -3.85410196624969670,  2.23606797749979291},
    {-0.06524758424985455,  0.61803398874989612, -1.00000000000000533},
    { 2.17082039324994058,  0.38196601125010903,  0.61803398874989645},
    { 0.78885438199982571, -1.00000000000000511,  0.38196601125010887}
   },
   {
    { 0.38315594803123498, -0.04774575140626385, -0.32725424859374058},
    { 0.05590169943749599,  0.37499999999999805, -0.04774575140626322}, 
    { 0.00815594803123186, -0.32725424859373486,  0.37499999999999805},
    { 1.34164078649987584,  0.00000000000000000,  0.00000000000000000},
    {-0.54270509831248481, -0.40450849718746778,  0.46352549156242151},
    {-0.07917960675006559, -0.05901699437494783, -0.40450849718747433}, 
    {-0.48368810393753652,  0.46352549156241651, -0.05901699437494128}, 
    { 0.26008130618755170, -1.96352549156242140,  0.25000000000000172},
    { 0.51008130618755820,  1.71352549156241984, -1.96352549156242140},
    {-1.45344418537486142,  0.25000000000000155,  1.71352549156241962}
   },
   {
    { 1.08541019662495430, -2.42705098312483658,  0.92705098312484435},
    {-0.49999999999999994, -0.49999999999999728,  0.80901699437494545},
    { 1.30901699437494079, -0.30901699437494822, -0.49999999999999744},
    {-0.44721359549995598,  0.00000000000000000,  0.00000000000000000},
    { 3.06524758424985500,  1.61803398874987980, -3.85410196624967849},
    {-1.00000000000000000,  1.00000000000000555, -0.38196601125010909},
    {-6.40688837074970685,  0.61803398874989656,  2.99999999999999556},
    { 1.17082039324994369,  0.61803398874989634, -1.00000000000000555},
    {-1.06524758424985544,  0.38196601125010920,  0.61803398874989645},
    { 2.78885438199982438, -1.00000000000000555,  0.38196601125010909}
   }
  }
  };

  int tNumGold_H=tGradientsGold.size();
  for(int h=0; h<tNumGold_H; h++)
  {
      int tNumGold_I=tGradientsGold[0].size();
      for(int i=0; i<tNumGold_I; i++)
      {
          int tNumGold_J=tGradientsGold[0][0].size();
          for(int j=0; j<tNumGold_J; j++)
          {
              int tNumGold_K=tGradientsGold[0][0][0].size();
              for(int k=0; k<tNumGold_K; k++)
              {
                  TEST_FLOATING_EQUALITY(tGradientsHost(h,i,j,k), tGradientsGold[h][i][j][k], 1e-13);
              }
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
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET10", meshWidth);

  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

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
  LAMBDA_EXPRESSION(int nodeOrdinal)
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
  LAMBDA_EXPRESSION(const int cellOrdinal, const int gpOrdinal)
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
  LAMBDA_EXPRESSION(int cellOrdinal)
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
{{22.43589743589807, -19.23076923076922, 21.50065362980549, 22.43589743589769, 9.615384615384512,
  11.88526901442095, -44.87179487179488, 9.615384615384709, 31.11603824519019, 0.000000000000000,
  0.000000000000000, 38.70117653365052, 44.87179487179349, -57.69230769230730, -47.06179991346048, 
  89.74358974359015, 19.23076923076904, -85.52333837499911, -134.6153846153846, 38.46153846153826,
 -27.83103068269109, 44.87179487179485, -76.92307692307710, 31.89152333654063, 134.6153846153850, 
  19.23076923076922, -25.80078435576689, -179.4871794871799, 57.69230769230789, 51.12229256730983},
 {72.60408923877753, -9.615384615384334, -9.615384615384889, 50.16819180287990, 19.23076923076924,
 -9.615384615384716, 27.73229436698222, -9.615384615384519, 19.23076923076923, 90.30274524518452,
  0.000000000000000, 0.000000000000000, -64.93907159294764, -38.46153846153865, -19.23076923076865,
 -109.8108664647411, 57.69230769230768, -38.46153846153863, -199.5544562083305, -19.23076923076942,
  57.69230769230768, 119.2853493237230, -57.69230769230787, -19.23076923076923, 74.41355445192896, 
  76.92307692307710, -57.69230769230789, -60.20183016345692, -19.23076923076924, 76.92307692307712},
 {50.16819180287946, 9.615384615384889, -19.23076923076921, 27.73229436698224, 9.615384615384711,
  9.615384615384517, 72.60408923877711, -19.23076923076924, 9.615384615384723, 90.30274524518454,
  0.000000000000000, 0.000000000000000, -109.8108664647411, 19.23076923076865, -57.69230769230730,
 -199.5544562083314, 38.46153846153865, 19.23076923076903, -64.93907159294582, -57.69230769230767, 
  38.46153846153825, 74.41355445192808, 19.23076923076923, -76.92307692307710, -60.20183016345604, 
  57.69230769230789, 19.23076923076923, 119.2853493237230, -76.92307692307712, 57.69230769230789},
 {-22.43589743589807, 31.11603824519037, -9.615384615384334, -22.43589743589767, 21.50065362980568,
  19.23076923076924, 44.87179487179490, 11.88526901442096, -9.615384615384524, 0.000000000000000, 
  38.70117653365052, 0.000000000000000, -44.87179487179353, -27.83103068269185, -38.46153846153864,
 -89.74358974359015, -47.06179991346049, 57.69230769230769, 134.6153846153846, -85.52333837499876,
 -19.23076923076943, -44.87179487179483, 51.12229256730985, -57.69230769230788, -134.6153846153851, 
  31.89152333654098, 76.92307692307712, 179.4871794871800, -25.80078435576725, -19.23076923076924},
{-44.87179487179484, 21.50065362980548, 9.615384615384889, 22.43589743589722, 11.88526901442096,
  9.615384615384698, 22.43589743589769, 31.11603824519018, -19.23076923076925, 0.000000000000000,
  38.70117653365051, 0.000000000000000, -134.6153846153837, -47.06179991346048, 19.23076923076867, 
  44.87179487179443, -85.52333837499916, 38.46153846153867, 89.74358974358924, -27.83103068269107,
 -57.69230769230768, -179.4871794871799, 31.89152333654061, 19.23076923076923, 44.87179487179483,
 -25.80078435576689, 57.69230769230789, 134.6153846153850, 51.12229256730986, -76.92307692307713},
{-22.43589743589675, -9.615384615384892, 31.11603824519036, 44.87179487179492, -9.615384615384720, 
  21.50065362980568, -22.43589743589721, 19.23076923076924, 11.88526901442095, 0.000000000000000,
  0.000000000000000, 38.70117653365054, -89.74358974359019, -19.23076923076862, -27.83103068269185, 
  134.6153846153845, -38.46153846153862, -47.06179991346045, -44.87179487179530, 57.69230769230765,
 -85.52333837499874, -134.6153846153851, -19.23076923076923, 51.12229256730984, 179.4871794871800,
 -57.69230769230790, 31.89152333654101, -44.87179487179483, 76.92307692307713, -25.80078435576730}};

  tNumGold_I=tResultGold.size();
  for(int iCell=0; iCell<tNumGold_I; iCell++)
  {
      int tNumGold_J=tResultGold[0].size();
      for(int j=0; j<tNumGold_J; j++)
      {
          if(tResultGold[iCell][j] == 0.0)
          {
              TEST_ASSERT(fabs(tResultHost(iCell,j)) < 1e-13);
          }
          else
          {
              TEST_FLOATING_EQUALITY(tResultHost(iCell,j), tResultGold[iCell][j], 1e-13);
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
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET10", meshWidth);

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
  LAMBDA_EXPRESSION(int nodeOrdinal)
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
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  using ElementType = typename Plato::Mechanics<Plato::Tet10>;

  Plato::DataMap tDataMap;
  Plato::Elliptic::VectorFunction<ElementType>
    tVectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto residual = tVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
 55.4645887339653996,  23.7705380288423100,  23.7705380288423029, -109.810866464739348, -85.5233383749995255,
 0.00000000000000000,  50.1681918028794414,  2.26988439903643524,  28.8461538461539639, -199.554456208332198,
 0.00000000000000000, -47.0617999134597511, -309.365322673071603,  0.00000000000000000,  0.00000000000000000,
-199.554456208331374,  38.4615384615386517,  19.2307692307690310,  5.29639693108502740,  28.8461538461539497,
 21.5006536298054733, -109.810866464741139,  57.6923076923076792, -38.4615384615386304,  55.4645887339644617,
 0.00000000000000000,  28.8461538461537508,  0.00000000000000000, -47.0617999134596943, -85.5233383749995255,
 0.00000000000000000, -132.585138288459234,  0.00000000000000000, -89.7435897435901495, -47.0617999134604901,
 57.6923076923076934,  0.00000000000000000,  0.00000000000000000, -132.585138288459234, -120.403660326913922,
-51.6015687115345258, -51.6015687115344974, -15.3300352916621705, -45.0315535865361340,  134.615384615385011,
 89.7435897435901495,  19.2307692307690417, -85.5233383749991134, -105.073625035250942,  134.615384615384983,
-6.57001512499805074, -120.403660326912956,  38.4615384615386446,  96.1538461538463451,  67.3076923076926050,
 21.5006536298054591,  2.26988439903642814,  44.8717948717944282, -85.5233383749991560,  38.4615384615386660,
 67.3076923076921219,  23.7705380288419157,  0.00000000000000000,  134.615384615384528, -38.4615384615386233,
-47.0617999134604474,  314.102564102564997, -6.57001512499804363, -45.0315535865361198,  224.358974358974791,
-51.6015687115341422,  38.4615384615386446,  0.00000000000000000,  28.8461538461537543,  23.7705380288418979,
 89.7435897435902064,  96.1538461538463594, -51.6015687115341919,  180.605490490369050,  77.4023530673010498,
 77.4023530673010498
  };

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

  std::vector<Plato::Scalar> gold_jac_entries = {
    722672.504457739997, -132494.267401092191, -132494.267401092162,
   -132494.267401092191,  722672.504457739880, -132494.267401092220,
   -132494.267401092220, -132494.267401092191,  722672.504457739997,
   -366349.067406447721, -3863.02002242947492,  217867.806893917208,
   -319.517429480503779, -133154.907763027761, -96117.0325768158218,
    233555.073531737318, -104717.294028738019, -380080.008596341766,
    77386.8667729836452, -2379.61224474191113, -72774.4549925839092,
   -8451.49426717760434,  36106.0357020234733,  41848.0239504697747,
   -73112.8293804046407,  41509.6495626489341,  101645.724934288271,
   -133154.907763027761, -96117.0325768158218, -319.517429480503779,
   -104717.294028738019, -380080.008596341766,  233555.073531737318,
   -3863.02002242947492,  217867.806893917208, -366349.067406447721
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
  }

#ifdef NOPE

  // compute and test gradient wrt control, z
  //
  auto gradient_z = tVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
-286.057692307692207, -199.519230769230717, -170.673076923076906,
-70.9134615384615188, -70.9134615384615188, 69.7115384615384528,
-56.4903846153845919, 84.1346153846153868, -56.4903846153845919,
-127.403846153846075, 13.2211538461538325, 13.2211538461538360
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


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
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET10", meshWidth);

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
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  using ElementType = typename Plato::Mechanics<Plato::Tet10>;

  Plato::DataMap tDataMap;
  Plato::Elliptic::VectorFunction<ElementType>
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
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET10", meshWidth);

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
    tElasticityProblem.readEssentialBoundaryConditions(*tParamList);


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

    std::vector<Plato::Scalar> tGold = {
3.18967e-7, 1.52576e-6, -5.08263e-9, 1.80438e-7,
1.6722e-6, -2.672e-8, 1.94494e-7, 1.48736e-6,
-1.10089e-8, 3.5485e-8, 1.4262e-6, -5.70112e-8,
2.62373e-7, 1.69246e-6, -1.45071e-7, 3.26086e-7,
1.32044e-6, -1.15541e-7, 2.0408e-7, 1.7353e-6,
-1.75332e-7, 1.69012e-7, 1.27502e-6, -1.06742e-7,
7.19454e-8
    };


    Plato::OrdinalType tDofOffset = 350; // comparing only the last 25 dofs
    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tDofIndex=0; tDofIndex < tGold.size(); tDofIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostSolution(tDofOffset+tDofIndex), tGold[tDofIndex], tTolerance);
    }
}

#ifdef NOPE

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
    "  <Parameter name='Objective' type='string' value='My Internal Elastic Energy'/> \n"
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
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);


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
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Internal Elastic Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution,z);

  Plato::Scalar value_gold = 48.15;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  tSolution.set("State", U);
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -1144.230769230769, -798.0769230769229, -682.6923076923076,
  -1427.884615384615, -1081.730769230769, -403.8461538461537,
  -283.6538461538461, -283.6538461538460,  278.8461538461538,
  -1370.192307692307, -461.5384615384613, -908.6538461538458,
  -2163.461538461537, -692.3076923076922, -576.9230769230769,
  -793.2692307692303, -230.7692307692308,  331.7307692307693,
  -225.9615384615384,  336.5384615384615, -225.9615384615383
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
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
   1.504687500000000,  2.006250000000001, 0.5015625000000001,
   2.006250000000001,  3.009375000000001, 1.003125000000000,
   0.5015625000000000, 1.003125000000000, 0.5015625000000001,
   2.006250000000001,  3.009375000000003, 1.003125000000001,
   3.009375000000003,  6.018750000000004, 3.009375000000002,
   1.003125000000001,  3.009375000000003, 2.006250000000001,
   0.5015625000000008, 1.003125000000001, 0.5015625000000010,
   1.003125000000002,  3.009375000000004, 2.006250000000002,
   0.5015625000000008, 2.006250000000004, 1.50468750000000
  };

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
10.16250000000000, 0.7125000000000001, -2.437500000000000,
9.713942307692305, -0.7745192307692330, 1.748076923076921,
-0.4485576923076922, -1.487019230769231, 4.185576923076923,
8.779326923076917, 4.932692307692306, -4.374519230769231,
6.499038461538459, 6.178846153846154, 2.059615384615383,
-2.280288461538460, 1.246153846153845, 6.434134615384615,
-1.383173076923076, 4.220192307692306, -1.937019230769230,
-3.214903846153847, 6.953365384615383, 0.3115384615384613,
-1.831730769230766, 2.733173076923077, 2.248557692307692,
11.99423076923077, -2.020673076923078, -4.686057692307688,
12.92884615384615, -7.727884615384617, 1.436538461538468,
0.9346153846153846, -5.707211538461538, 6.122596153846152,
11.05961538461539, 3.686538461538455, -10.80865384615385,
 0.00000000000000, 0.000000000000000, 0.0000000000000000
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-13);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}

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
  auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", meshWidth);


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
  Plato::DataMap dataMap;
  std::string tMyFunction("Globalized Stress");
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


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
