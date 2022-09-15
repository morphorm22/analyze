#ifndef APPLY_WEIGHTING_HPP
#define APPLY_WEIGHTING_HPP

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Weighting functor.

 Given an input view and density, apply weighting to the input view.
 Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType NumNodes, Plato::OrdinalType NumTerms, typename PenaltyFunction>
class ApplyWeighting
{
private:
    PenaltyFunction mPenaltyFunction; /*!< penalty model used for topology optimization - density discretization */

public:
    /******************************************************************************//**
     * \brief Default Constructor
     * \param [in] aPenaltyFunction penalty function interface
    **********************************************************************************/
    ApplyWeighting(PenaltyFunction aPenaltyFunction) :
        mPenaltyFunction(aPenaltyFunction)
    {
    }

    /******************************************************************************//**
     * \brief Update penalty model parameters within a frequency of optimization iterations
    **********************************************************************************/
    void update()
    {
        mPenaltyFunction.update();
    }

    /******************************************************************************//**
     * \brief Evaluate penalty model
     * \param [in] aCellOrdinal cell/element ordinal
     * \param [in] aInputOutput penalized 2D view
     * \param [in] aControl     control, i.e. design, variables
    **********************************************************************************/
    template<typename InputScalarType, typename WeightScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
              Plato::OrdinalType                            aCellOrdinal,
        const Plato::ScalarMultiVectorT<WeightScalarType> & aControl,
        const Plato::Array<NumNodes>                      & aBasisValues,
              Plato::Array<NumTerms, InputScalarType>     & aInputOutput
    ) const
    {
        // apply weighting
        //
        WeightScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode)*aBasisValues(tNode);
        }
        for (Plato::OrdinalType tTerm = 0; tTerm < NumTerms; tTerm++)
        {
            aInputOutput(tTerm) *= mPenaltyFunction(tCellDensity);
        }
    }

    /******************************************************************************//**
     * \brief Evaluate penalty model
     * \param [in] aCellOrdinal cell/element ordinal
     * \param [in] aGpOrdinal gauss point ordinal
     * \param [in] aInputOutput penalized 2D view
     * \param [in] aControl     control, i.e. design, variables
    **********************************************************************************/
    template<typename InputScalarType, typename WeightScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
              Plato::OrdinalType                            aCellOrdinal,
              Plato::OrdinalType                            aGpOrdinal,
        const Plato::ScalarMultiVectorT<WeightScalarType> & aControl,
        const Plato::Array<NumNodes>                      & aBasisValues,
              Plato::ScalarArray3DT<InputScalarType>        aInputOutput
    ) const
    {
        // apply weighting
        //
        WeightScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode)*aBasisValues(tNode);
        }
        for (Plato::OrdinalType tTerm = 0; tTerm < NumTerms; tTerm++)
        {
            aInputOutput(aCellOrdinal, aGpOrdinal, tTerm) *= mPenaltyFunction(tCellDensity);
        }
    }


    /******************************************************************************//**
     * \brief Evaluate penalty model
     * \param [in] aCellOrdinal cell/element ordinal
     * \param [in] aInputOutput penalized 2D view
     * \param [in] aControl     control, i.e. design, variables
    **********************************************************************************/
    template<typename InputScalarType, typename WeightScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
              Plato::OrdinalType                            aCellOrdinal,
        const Plato::ScalarMultiVectorT<WeightScalarType> & aControl,
        const Plato::Array<NumNodes>                      & aBasisValues,
              InputScalarType                             & aInputOutput
    ) const
    {
        // apply weighting
        //
        WeightScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode)*aBasisValues(tNode);
        }
        aInputOutput *= mPenaltyFunction(tCellDensity);
    }

    /******************************************************************************//**
     * \brief Evaluate penalty model
     * \param [in] aCellOrdinal cell/element ordinal
     * \param [in] aInputOutput penalized 2D view
     * \param [in] aControl     control, i.e. design, variables
    **********************************************************************************/
    template<typename InputScalarType, typename WeightScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::ScalarMultiVectorT<InputScalarType>  const & aInputOutput,
        Plato::ScalarMultiVectorT<WeightScalarType> const & aControl
    ) const
    {
        // apply weighting
        //
        WeightScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode);
        }
        tCellDensity = (tCellDensity / NumNodes);
        for (Plato::OrdinalType tTerm = 0; tTerm < NumTerms; tTerm++)
        {
            aInputOutput(aCellOrdinal, tTerm) *= mPenaltyFunction(tCellDensity);
        }
    }

    /******************************************************************************//**
     * \brief Evaluate penalty model
     * \param [in] aCellOrdinal cell/element ordinal
     * \param [in] aInput       input 2D view
     * \param [in] aOutput      penalized 2D view
     * \param [in] aControl     control, i.e. design, variables
    **********************************************************************************/
    template<typename InputScalarType, typename OutputScalarType, typename WeightScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::ScalarMultiVectorT<InputScalarType>  const & aInput,
        Plato::ScalarMultiVectorT<OutputScalarType> const & aOutput,
        Plato::ScalarMultiVectorT<WeightScalarType> const & aControl
    ) const
    {
        // apply weighting
        //
        WeightScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode);
        }
        tCellDensity = (tCellDensity / NumNodes);
        for (Plato::OrdinalType tTerm = 0; tTerm < NumTerms; tTerm++)
        {
            aOutput(aCellOrdinal, tTerm) = mPenaltyFunction(tCellDensity) * aInput(aCellOrdinal, tTerm);
        }
    }

    /******************************************************************************//**
     * \brief Evaluate penalty model
     * \param [in] aCellOrdinal cell/element ordinal
     * \param [in] aResult      penalized 1D view
     * \param [in] aControl     control, i.e. design, variables
    **********************************************************************************/
    template<typename ResultScalarType, typename WeightScalarType>
    KOKKOS_INLINE_FUNCTION
    void operator()(Plato::OrdinalType aCellOrdinal,
                    Plato::ScalarVectorT<ResultScalarType> const &aResult,
                    Plato::ScalarMultiVectorT<WeightScalarType> const &aControl) const
    {
        // apply weighting
        //
        WeightScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode);
        }
        tCellDensity = (tCellDensity / NumNodes);
        aResult(aCellOrdinal) *= mPenaltyFunction(tCellDensity);
    }

    /******************************************************************************//**
     * \brief Evaluate penalty model
     * \param [in] aCellOrdinal cell/element ordinal
     * \param [in] aInput       input 1D view
     * \param [in] aOutput      penalized 1D view
     * \param [in] aControl     control, i.e. design, variables
    **********************************************************************************/
    template<typename InputScalarType, typename OutputScalarType, typename WeightScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarVectorT<InputScalarType> const &aInput,
               Plato::ScalarVectorT<OutputScalarType> &aOutput,
               Plato::ScalarMultiVectorT<WeightScalarType> const &aControl) const
    {
        // apply weighting
        //
        WeightScalarType tCellDensity = 0.0;
        for (Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
        {
            tCellDensity += aControl(aCellOrdinal, tNode);
        }
        tCellDensity = (tCellDensity / NumNodes);
        aOutput(aCellOrdinal) = mPenaltyFunction(tCellDensity) * aInput(aCellOrdinal);
    }
};
// class ApplyWeighting

}
// namespace Plato

#endif
