#include "hyperbolic/micromorphic/CubicMicromorphicInertiaMaterial.hpp"

namespace Plato
{

template<>
Plato::CubicMicromorphicInertiaMaterial<1>::
CubicMicromorphicInertiaMaterial(const Teuchos::ParameterList& paramList) :
    Plato::MicromorphicInertiaMaterial<1>(paramList)
{
    Plato::Scalar tRho = paramList.get<Plato::Scalar>("Mass Density");
    mCellMacroscopicDensity = tRho;    

    Plato::Scalar tEta_bar_1 = paramList.get<Plato::Scalar>("Eta_bar_1");
    Plato::Scalar tEta_bar_3 = paramList.get<Plato::Scalar>("Eta_bar_3");
    mCellInertiaTe(0, 0)     = tEta_bar_3 + 2.0 * tEta_bar_1;

    Plato::Scalar tEta_bar_2 = paramList.get<Plato::Scalar>("Eta_bar_2");
    mCellInertiaTc(0, 0)     = tEta_bar_2;

    Plato::Scalar tEta_1 = paramList.get<Plato::Scalar>("Eta_1");
    Plato::Scalar tEta_3 = paramList.get<Plato::Scalar>("Eta_3");
    mCellInertiaJm(0, 0) = tEta_3 + 2.0 * tEta_1;

    Plato::Scalar tEta_2 = paramList.get<Plato::Scalar>("Eta_2");
    mCellInertiaJc(0, 0) = tEta_2;
}

template<>
Plato::CubicMicromorphicInertiaMaterial<2>::
CubicMicromorphicInertiaMaterial(const Teuchos::ParameterList& paramList) :
    Plato::MicromorphicInertiaMaterial<2>(paramList)
{
    Plato::Scalar tRho = paramList.get<Plato::Scalar>("Mass Density");
    mCellMacroscopicDensity = tRho;    

    Plato::Scalar tEta_bar_1      = paramList.get<Plato::Scalar>("Eta_bar_1");
    Plato::Scalar tEta_bar_3      = paramList.get<Plato::Scalar>("Eta_bar_3");
    Plato::Scalar tEta_bar_star_1 = paramList.get<Plato::Scalar>("Eta_bar_star_1");
    mCellInertiaTe(0, 0)          = tEta_bar_3 + 2.0 * tEta_bar_1;
    mCellInertiaTe(0, 1)          = tEta_bar_3;
    mCellInertiaTe(1, 0)          = tEta_bar_3;
    mCellInertiaTe(1, 1)          = tEta_bar_3 + 2.0 * tEta_bar_1;
    mCellInertiaTe(2, 2)          = tEta_bar_star_1;

    Plato::Scalar tEta_bar_2 = paramList.get<Plato::Scalar>("Eta_bar_2");
    mCellInertiaTc(0, 0)     = tEta_bar_2;

    Plato::Scalar tEta_1      = paramList.get<Plato::Scalar>("Eta_1");
    Plato::Scalar tEta_3      = paramList.get<Plato::Scalar>("Eta_3");
    Plato::Scalar tEta_star_1 = paramList.get<Plato::Scalar>("Eta_star_1");
    mCellInertiaJm(0, 0)      = tEta_3 + 2.0 * tEta_1;
    mCellInertiaJm(0, 1)      = tEta_3;
    mCellInertiaJm(1, 0)      = tEta_3;
    mCellInertiaJm(1, 1)      = tEta_3 + 2.0 * tEta_1;
    mCellInertiaJm(2, 2)      = tEta_star_1;

    Plato::Scalar tEta_2 = paramList.get<Plato::Scalar>("Eta_2");
    mCellInertiaJc(0, 0) = tEta_2;
}

template<>
Plato::CubicMicromorphicInertiaMaterial<3>::
CubicMicromorphicInertiaMaterial(const Teuchos::ParameterList& paramList) :
    Plato::MicromorphicInertiaMaterial<3>(paramList)
{
    Plato::Scalar tRho = paramList.get<Plato::Scalar>("Mass Density");
    mCellMacroscopicDensity = tRho;    

    Plato::Scalar tEta_bar_1      = paramList.get<Plato::Scalar>("Eta_bar_1");
    Plato::Scalar tEta_bar_3      = paramList.get<Plato::Scalar>("Eta_bar_3");
    Plato::Scalar tEta_bar_star_1 = paramList.get<Plato::Scalar>("Eta_bar_star_1");
    mCellInertiaTe(0, 0)          = tEta_bar_3 + 2.0 * tEta_bar_1;
    mCellInertiaTe(0, 1)          = tEta_bar_3;
    mCellInertiaTe(0, 2)          = tEta_bar_3;
    mCellInertiaTe(1, 0)          = tEta_bar_3;
    mCellInertiaTe(1, 1)          = tEta_bar_3 + 2.0 * tEta_bar_1;
    mCellInertiaTe(1, 2)          = tEta_bar_3;
    mCellInertiaTe(2, 0)          = tEta_bar_3;
    mCellInertiaTe(2, 1)          = tEta_bar_3;
    mCellInertiaTe(2, 2)          = tEta_bar_3 + 2.0 * tEta_bar_1;
    mCellInertiaTe(3, 3)          = tEta_bar_star_1;
    mCellInertiaTe(4, 4)          = tEta_bar_star_1;
    mCellInertiaTe(5, 5)          = tEta_bar_star_1;

    Plato::Scalar tEta_bar_2 = paramList.get<Plato::Scalar>("Eta_bar_2");
    mCellInertiaTc(0, 0)     = tEta_bar_2;
    mCellInertiaTc(1, 1)     = tEta_bar_2;
    mCellInertiaTc(2, 2)     = tEta_bar_2;

    Plato::Scalar tEta_1      = paramList.get<Plato::Scalar>("Eta_1");
    Plato::Scalar tEta_3      = paramList.get<Plato::Scalar>("Eta_3");
    Plato::Scalar tEta_star_1 = paramList.get<Plato::Scalar>("Eta_star_1");
    mCellInertiaJm(0, 0)      = tEta_3 + 2.0 * tEta_1;
    mCellInertiaJm(0, 1)      = tEta_3;
    mCellInertiaJm(0, 2)      = tEta_3;
    mCellInertiaJm(1, 0)      = tEta_3;
    mCellInertiaJm(1, 1)      = tEta_3 + 2.0 * tEta_1;
    mCellInertiaJm(1, 2)      = tEta_3;
    mCellInertiaJm(2, 0)      = tEta_3;
    mCellInertiaJm(2, 1)      = tEta_3;
    mCellInertiaJm(2, 2)      = tEta_3 + 2.0 * tEta_1;
    mCellInertiaJm(3, 3)      = tEta_star_1;
    mCellInertiaJm(4, 4)      = tEta_star_1;
    mCellInertiaJm(5, 5)      = tEta_star_1;

    Plato::Scalar tEta_2 = paramList.get<Plato::Scalar>("Eta_2");
    mCellInertiaJc(0, 0) = tEta_2;
    mCellInertiaJc(1, 1) = tEta_2;
    mCellInertiaJc(2, 2) = tEta_2;
}

}
