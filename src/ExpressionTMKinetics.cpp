/*
 * ExpressionTMKinetics.cpp
 *
 */

#include "ExpressionTMKinetics.hpp"

/*

    //template<typename EvaluationType, typename SimplexPhysics>
    void ExpressionTMKinetics::perform_expression_call()
    {
        printf("Local Control Values\n");
        Plato::ScalarMultiVectorT<ControlScalarType> tLocalControl("Local Control", 20, 4);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,20), LAMBDA_EXPRESSION(Plato::OrdinalType i)
        {
            for(int j=0; j<4; ++j)
            {
                tLocalControl(i, j) = .5;
                printf("%lf ", tLocalControl(i, j));
            }
            printf("\n");
        },"Set Local Control");


        Plato::ScalarVectorT<ControlScalarType> tElementDensity("Gauss point density", 20);
        printf("Element Density Values\n");
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,20), LAMBDA_EXPRESSION(Plato::OrdinalType i)
        {
            tElementDensity(i) = .5;
            printf("%lf\n", tElementDensity(i));
        },"Compute Element Densities");
    
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementYoungsModulusValues("Element Youngs Modulus", 20, 1);

        ExpressionEvaluator<Plato::ScalarMultiVectorT<KineticsScalarType>,
                            Plato::ScalarMultiVectorT<KinematicsScalarType>,
                            Plato::ScalarVectorT<ControlScalarType>,
                            Plato::Scalar > tExpEval;
        
        tExpEval.parse_expression("E0*tElementDensity*tElementDensity*tElementDensity");
        tExpEval.setup_storage(20, 1);
        tExpEval.set_variable("E0", 1e9);
        printf("Element Youngs Modulus Values\n");
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,20), LAMBDA_EXPRESSION(Plato::OrdinalType i)
        {
            tExpEval.set_variable("tElementDensity", tElementDensity, i);
            std::stringstream tMessage;
            tExpEval.print_variables(tMessage);
            printf("%s\n", tMessage.str().c_str());
            tExpEval.evaluate_expression( i, tElementYoungsModulusValues );
            printf("%lf\n", tElementYoungsModulusValues(i,0));
        },"Compute Youngs Modulus for each Element");
        Kokkos::fence();
        tExpEval.clear_storage();

    }
*/


/*
#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF2(Plato::ExpressionTMKinetics  , Plato::SimplexThermomechanics, 3)
#endif
*/
