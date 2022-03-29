#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoStaticsTypes.hpp"
#include "MechanicsElement.hpp"
#include "Tet10.hpp"
#include "Tri6.hpp"

#include "SurfaceArea.hpp"

using ordType = typename Plato::ScalarMultiVector::size_type;


/******************************************************************************/
/*! 
  \brief Check the Tet10 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, Tet10_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Tet10::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Tet10::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Tet10::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 10);
    TEST_ASSERT(tNodesPerFace  == 6 );
    TEST_ASSERT(tSpaceDims     == 3 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Tet10> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, MechanicsTet10_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 10);
    TEST_ASSERT(tNodesPerFace  == 6 );
    TEST_ASSERT(tDofsPerCell   == 30);
    TEST_ASSERT(tDofsPerNode   == 3 );
    TEST_ASSERT(tSpaceDims     == 3 );
    TEST_ASSERT(tNumVoigtTerms == 6 );
}

/******************************************************************************/
/*! 
  \brief Check the Tri6 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, Tri6_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Tri6::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Tri6::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Tri6::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 6 );
    TEST_ASSERT(tNodesPerFace  == 3 );
    TEST_ASSERT(tSpaceDims     == 2 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Tri6> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, MechanicsTri6_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tri6>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 6 );
    TEST_ASSERT(tNodesPerFace  == 3 );
    TEST_ASSERT(tDofsPerCell   == 12);
    TEST_ASSERT(tDofsPerNode   == 2 );
    TEST_ASSERT(tSpaceDims     == 2 );
    TEST_ASSERT(tNumVoigtTerms == 3 );
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tet10 basis functions at (0.25, 0.25, 0.25) and at
         each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", 11, Plato::Tet10::mNumNodesPerCell);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,1), LAMBDA_EXPRESSION(int ordinal)
    {
        Plato::Array<Plato::Tet10::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        auto tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(0,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.0; tPoint(2) = 0.0;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(1,i) = tValues(i); }

        tPoint(0) = 1.0; tPoint(1) = 0.0; tPoint(2) = 0.0;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(2,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 1.0; tPoint(2) = 0.0;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(3,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.0; tPoint(2) = 1.0;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(4,i) = tValues(i); }

        tPoint(0) = 0.5; tPoint(1) = 0.0; tPoint(2) = 0.0;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(5,i) = tValues(i); }

        tPoint(0) = 0.5; tPoint(1) = 0.5; tPoint(2) = 0.0;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(6,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.5; tPoint(2) = 0.0;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(7,i) = tValues(i); }

        tPoint(0) = 0.5; tPoint(1) = 0.0; tPoint(2) = 0.5;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(8,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.5; tPoint(2) = 0.5;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(9,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.0; tPoint(2) = 0.5;
        tValues = Plato::Tet10::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++) { tValuesView(10,i) = tValues(i); }


    }, "basis functions");

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    std::vector<std::vector<Plato::Scalar>> tValuesGold = {
      {-1.0/8.0, -1.0/8.0, -1.0/8.0, -1.0/8.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0},
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    int tNumGold_I=tValuesGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tValuesGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tValuesHost(i,j), tValuesGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tri6 basis functions at (1/6, 1/6) and at
         each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", 7, Plato::Tri6::mNumNodesPerCell);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,1), LAMBDA_EXPRESSION(int ordinal)
    {
        Plato::Array<Plato::Tri6::mNumSpatialDims> tPoint;

        tPoint(0) = Plato::Scalar(1)/6; tPoint(1) = Plato::Scalar(1)/6;
        auto tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(0,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(1,i) = tValues(i); }

        tPoint(0) = 1.0; tPoint(1) = 0.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(2,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 1.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(3,i) = tValues(i); }

        tPoint(0) = 0.5; tPoint(1) = 0.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(4,i) = tValues(i); }

        tPoint(0) = 0.5; tPoint(1) = 0.5;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(5,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.5;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(6,i) = tValues(i); }

    }, "basis functions");

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    std::vector<std::vector<Plato::Scalar>> tValuesGold = {
      {2.0/9.0, -1.0/9.0, -1.0/9.0, 4.0/9.0, 1.0/9.0, 4.0/9.0},
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    int tNumGold_I=tValuesGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tValuesGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tValuesHost(i,j), tValuesGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tet10 basis function gradients at (0.25, 0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumSpatialDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,1), LAMBDA_EXPRESSION(int ordinal)
    {
        Plato::Array<Plato::Tet10::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        auto tGrads = Plato::Tet10::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Tet10::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    }, "basis function derivatives");

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
      {0, 0, 0}, { 0, 0, 0 }, {0, 0, 0}, {0, 0, 0}, { 0, -1, -1},
      {1, 1, 0}, {-1, 0, -1}, {1, 0, 1}, {0, 1, 1}, {-1, -1,  0}
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tri6 basis function gradients at (1/6, 1/6)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Tri6::mNumNodesPerCell, Plato::Tri6::mNumSpatialDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,1), LAMBDA_EXPRESSION(int ordinal)
    {
        Plato::Array<Plato::Tri6::mNumSpatialDims> tPoint;

        tPoint(0) = Plato::Scalar(1)/6; tPoint(1) = Plato::Scalar(1)/6;
        auto tGrads = Plato::Tri6::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Tri6::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    }, "basis function derivatives");

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
        {Plato::Scalar(-5)/3, Plato::Scalar(-5)/3},
        {Plato::Scalar(-1)/3, 0                  },
        {0,                   Plato::Scalar(-1)/3},
        {2,                   Plato::Scalar(-2)/3},
        {Plato::Scalar(2)/3,  Plato::Scalar(2)/3 },
        {Plato::Scalar(-2)/3, 2                  }
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tet10 jacobian at (0.25, 0.25, 0.25) for a cell that's
  in the reference configuration.  Jacobian should be identity.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, JacobianParentCoords )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    Plato::ScalarMultiVector tJacobianView("jacobian", Plato::Tet10::mNumSpatialDims, Plato::Tet10::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("node locations", 1, Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumSpatialDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,1), LAMBDA_EXPRESSION(int ordinal)
    {
        tConfig(0,0,0) = 0.0; tConfig(0,0,1) = 0.0; tConfig(0,0,2) = 0.0;
        tConfig(0,1,0) = 1.0; tConfig(0,1,1) = 0.0; tConfig(0,1,2) = 0.0;
        tConfig(0,2,0) = 0.0; tConfig(0,2,1) = 1.0; tConfig(0,2,2) = 0.0;
        tConfig(0,3,0) = 0.0; tConfig(0,3,1) = 0.0; tConfig(0,3,2) = 1.0;
        tConfig(0,4,0) = 0.5; tConfig(0,4,1) = 0.0; tConfig(0,4,2) = 0.0;
        tConfig(0,5,0) = 0.5; tConfig(0,5,1) = 0.5; tConfig(0,5,2) = 0.0;
        tConfig(0,6,0) = 0.0; tConfig(0,6,1) = 0.5; tConfig(0,6,2) = 0.0;
        tConfig(0,7,0) = 0.5; tConfig(0,7,1) = 0.0; tConfig(0,7,2) = 0.5;
        tConfig(0,8,0) = 0.0; tConfig(0,8,1) = 0.5; tConfig(0,8,2) = 0.5;
        tConfig(0,9,0) = 0.0; tConfig(0,9,1) = 0.0; tConfig(0,9,2) = 0.5;

        //Plato::Array<Plato::Tet10::mNumSpatialDims> tPoint;
        Plato::Array<ElementType::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        //auto tJacobian = Plato::Tet10::jacobian(tPoint, tConfig, ordinal);
        auto tJacobian = ElementType::jacobian(tPoint, tConfig, ordinal);
        for(ordType i=0; i<Plato::Tet10::mNumSpatialDims; i++)
        {
            for(ordType j=0; j<Plato::Tet10::mNumSpatialDims; j++)
            {
                tJacobianView(i,j) = tJacobian(i,j);
            }
        }
    }, "cell jacobian");

    auto tJacobianHost = Kokkos::create_mirror_view( tJacobianView );
    Kokkos::deep_copy( tJacobianHost, tJacobianView );

    std::vector<std::vector<Plato::Scalar>> tJacobianGold = { {1, 0, 0}, { 0, 1, 0 }, {0, 0, 1} };

    int tNumGold_I=tJacobianGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tJacobianGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tJacobianHost(i,j), tJacobianGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tri6 jacobian at (1/6, 1/6) for a cell that's
  in the reference configuration.  Jacobian should be identity.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, JacobianParentCoords )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tri6>;

    Plato::ScalarMultiVector tJacobianView("jacobian", Plato::Tri6::mNumSpatialDims, Plato::Tri6::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("node locations", 1, Plato::Tri6::mNumNodesPerCell, Plato::Tri6::mNumSpatialDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,1), LAMBDA_EXPRESSION(int ordinal)
    {
        tConfig(0,0,0) = 0.0; tConfig(0,0,1) = 0.0;
        tConfig(0,1,0) = 1.0; tConfig(0,1,1) = 0.0;
        tConfig(0,2,0) = 0.0; tConfig(0,2,1) = 1.0;
        tConfig(0,3,0) = 0.5; tConfig(0,3,1) = 0.0;
        tConfig(0,4,0) = 0.5; tConfig(0,4,1) = 0.5;
        tConfig(0,5,0) = 0.0; tConfig(0,5,1) = 0.5;

        Plato::Array<ElementType::mNumSpatialDims> tPoint;

        tPoint(0) = Plato::Scalar(1)/6; tPoint(1) = Plato::Scalar(1)/6;
        auto tJacobian = ElementType::jacobian(tPoint, tConfig, ordinal);
        for(ordType i=0; i<Plato::Tri6::mNumSpatialDims; i++)
        {
            for(ordType j=0; j<Plato::Tri6::mNumSpatialDims; j++)
            {
                tJacobianView(i,j) = tJacobian(i,j);
            }
        }
    }, "cell jacobian");

    auto tJacobianHost = Kokkos::create_mirror_view( tJacobianView );
    Kokkos::deep_copy( tJacobianHost, tJacobianView );

    std::vector<std::vector<Plato::Scalar>> tJacobianGold = {{1, 0}, {0, 1}};

    int tNumGold_I=tJacobianGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tJacobianGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tJacobianHost(i,j), tJacobianGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the surface area of a Tet10 face in the reference
         configuration.  Should evaluate to 1/2.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, SurfaceArea )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    Plato::SurfaceArea<ElementType> surfaceArea;

    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tCubaturePoints  = ElementType::Face::getCubPoints();
    auto tNumPoints = tCubatureWeights.size();

    constexpr auto tNodesPerFace = ElementType::mNumNodesPerFace;

    Plato::OrdinalVector tNodeOrds("node ordinals", Plato::Tet10::mNumNodesPerFace);

    Plato::ScalarMultiVector tJacobianView("jacobian", Plato::Tet10::mNumSpatialDims, Plato::Tet10::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("node locations", 1, Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumSpatialDims);

    Plato::ScalarVector tSurfaceArea("area at GP", tNumPoints);
    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{1, tNumPoints}),
    LAMBDA_EXPRESSION(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)

    {
        tConfig(0,0,0) = 0.0; tConfig(0,0,1) = 0.0; tConfig(0,0,2) = 0.0;
        tConfig(0,1,0) = 1.0; tConfig(0,1,1) = 0.0; tConfig(0,1,2) = 0.0;
        tConfig(0,2,0) = 0.0; tConfig(0,2,1) = 1.0; tConfig(0,2,2) = 0.0;
        tConfig(0,3,0) = 0.0; tConfig(0,3,1) = 0.0; tConfig(0,3,2) = 1.0;
        tConfig(0,4,0) = 0.5; tConfig(0,4,1) = 0.0; tConfig(0,4,2) = 0.0;
        tConfig(0,5,0) = 0.5; tConfig(0,5,1) = 0.5; tConfig(0,5,2) = 0.0;
        tConfig(0,6,0) = 0.0; tConfig(0,6,1) = 0.5; tConfig(0,6,2) = 0.0;
        tConfig(0,7,0) = 0.5; tConfig(0,7,1) = 0.0; tConfig(0,7,2) = 0.5;
        tConfig(0,8,0) = 0.0; tConfig(0,8,1) = 0.5; tConfig(0,8,2) = 0.5;
        tConfig(0,9,0) = 0.0; tConfig(0,9,1) = 0.0; tConfig(0,9,2) = 0.5;

        tNodeOrds(0) = 0; tNodeOrds(1) = 1; tNodeOrds(2) = 2;
        tNodeOrds(3) = 4; tNodeOrds(4) = 5; tNodeOrds(5) = 6;

        Plato::Array<Plato::Tet10::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
        for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<tNodesPerFace; tNodeOrd++)
        {
            tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*tNodesPerFace+tNodeOrd);
        }

        auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
        auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
        auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
        auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

        Plato::Scalar tSurfaceAreaGP(0.0);
        surfaceArea(aSideOrdinal, tLocalNodeOrds, tBasisGrads, tConfig, tSurfaceAreaGP);

        tSurfaceArea(aPointOrdinal) = tSurfaceAreaGP*tCubatureWeight;

    }, "face area");

    auto tAreasHost = Kokkos::create_mirror_view( tSurfaceArea );
    Kokkos::deep_copy( tAreasHost, tSurfaceArea );

    std::vector<Plato::Scalar> tAreasGold = {
        Plato::Scalar(1)/6, Plato::Scalar(1)/6, Plato::Scalar(1)/6
    };

    int tNumGold_I=tAreasGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        TEST_FLOATING_EQUALITY(tAreasHost(i), tAreasGold[i], 1e-13);
    }

}
