#include "EngineMeshIO.hpp"

#include "PlatoUtilities.hpp"
#include "AnalyzeAppUtils.hpp"

namespace Plato
{
    std::string strint(std::string aBase, Plato::OrdinalType aIndex)
    {
        std::stringstream ss;
        ss << aBase;
        ss << "_" << aIndex;
        return ss.str();
    }

    void checkAddExtension(std::string & aName, std::string aExt)
    {
        auto found = aName.find(aExt);
        if( found != (aName.size() - aExt.size()) )
        {
            aName += aExt;
        }
    }

    EngineMeshIO::EngineMeshIO(
        std::string         aName,
        Plato::EngineMesh & aMesh,
        std::string         aMode
    ) :
        mMesh(aMesh),
        mVariablesAreSet(false),
        mPlotIndex(0)
    {
        mMeshIO = std::make_shared<ExodusIO>();
        
        ::MeshIO::Type tMode;
        auto tModeLower = Plato::tolower(aMode);
        if( tModeLower == "write"  || tModeLower == "w") { tMode = ::MeshIO::CLOBBER; }
        else
        if( tModeLower == "read"   || tModeLower == "r") { tMode = ::MeshIO::READ; }
        else
        if( tModeLower == "append" || tModeLower == "a") { tMode = ::MeshIO::WRITE; }

        checkAddExtension(aName, ".exo");

        mMeshIO->initMeshIO( mMesh.mMesh.get(), mMesh.mDataContainer.get(), aName.c_str(), tMode );

        mMeshIO->openMeshIO();

        setVariableTypeSuffixes();

        if( tMode == ::MeshIO::CLOBBER || tMode == ::MeshIO::WRITE )
        {
            mMeshIO->writePrologue();
        }

        if( tMode == ::MeshIO::READ )
        {
            parseNodeVarNames();
        }
    }

    void
    EngineMeshIO::parseNodeVarNames()
    {
        auto tNodeVarNames = mMeshIO->getNodeVarNames();

        auto tNumSpaceDims = mMesh.NumDimensions();

        // find tensors

        // find symtensors
        
        // find vectors
        std::multimap<std::string, std::pair<std::string, std::string>> tBaseToExt;
        for( const auto& tNodeVarName : tNodeVarNames )
        {
            std::string tBase(tNodeVarName);
            std::string tExt(1, tBase.back());
            tBase.pop_back(); // remove label ('X', 'Y', etc.)
            if(tBase.back() == ' ') tBase.pop_back(); // remove space if present
            tBaseToExt.insert({tBase, {tExt, tNodeVarName}});
        }

        std::vector<std::string> tUniqueKeys;
        for( auto tIter = tBaseToExt.begin(), end = tBaseToExt.end(); tIter != end; tIter = tBaseToExt.upper_bound(tIter->first) )
        {
            tUniqueKeys.push_back(tIter->first);
        }

        for( const auto& tUniqueKey : tUniqueKeys )
        {
            if( tBaseToExt.count(tUniqueKey) == tNumSpaceDims )
            {
                std::vector<std::string> tTermNames;
                for( const auto& tVectorSuffix : mVectorSuffixes )
                {
                    auto tRange = tBaseToExt.equal_range(tUniqueKey);
                    for(auto tIter = tRange.first; tIter != tRange.second; ++tIter)
                    {
                        auto tExt = tIter->second.first;
                        if( tExt == tVectorSuffix.first )
                        {
                            tTermNames.push_back(tIter->second.second);
                        }
                    }
                }
                if( tTermNames.size() == tNumSpaceDims )
                {
                    mVectorNodeVars[tUniqueKey] = tTermNames;
                }
            }
        }
    }

    void
    EngineMeshIO::setVariableTypeSuffixes()
    {
        auto tNumSpaceDims = mMesh.NumDimensions();
        
        mScalarSuffixes = PairList({{"", 0}});
        if( tNumSpaceDims == 1 )
        {
            mVectorSuffixes = PairList({{"X", 0}});
            mSymtensorSuffixes = PairList({{"XX", 0}});
        }
        else
        if( tNumSpaceDims == 2 )
        {
            mVectorSuffixes = PairList({{"X", 0}, {"Y", 1}});
            mSymtensorSuffixes = PairList({{"XX", 0}, {"YY", 1}, {"XY", 2}});
        }
        else
        if( tNumSpaceDims == 3 )
        {
            mVectorSuffixes = PairList({{"X", 0}, {"Y", 1}, {"Z", 2}});
            mSymtensorSuffixes = PairList({{"XX", 0}, {"YY", 1}, {"ZZ", 2}, {"XY", 5}, {"XZ", 4}, {"YZ", 3}});
        }
        else
        {
            ANALYZE_THROWERR("EngineMeshIO: Invalid number of space dimensions.");
        }
    }

    EngineMeshIO::~EngineMeshIO()
    {
    }

    Plato::OrdinalType EngineMeshIO::NumNodes()    const {return mMesh.NumNodes();}
    Plato::OrdinalType EngineMeshIO::NumElements() const {return mMesh.NumElements();}

    void EngineMeshIO::AddNodeData(
        std::string         aName,
        Plato::ScalarVector aData
    )
    {
        AddNodeData(aName, aData, {aName});
    }
    void EngineMeshIO::AddNodeData(
        std::string         aName,
        Plato::ScalarVector aData,
        Plato::OrdinalType  aNumDofs
    )
    {
        auto tNumDims = mMesh.NumDimensions();
        if(aNumDofs == 1) // scalar
        {
            AddNodeData(aName, aData, {aName});
        }
        else
        if(aNumDofs == tNumDims) // vector
        {
            StrList tDofNames;
            tDofNames.push_back(aName+" X");
            if( tNumDims > 1 )  tDofNames.push_back(aName+" Y");
            if( tNumDims > 2 )  tDofNames.push_back(aName+" Z");
            AddNodeData(aName, aData, tDofNames);
        }
        else // just number them
        {
            StrList tDofNames;
            for( decltype(aNumDofs) iDof=0; iDof<aNumDofs; iDof++)
            {
                tDofNames.push_back(strint(aName, iDof));
            }
            AddNodeData(aName, aData, tDofNames);
        }
    }
    void EngineMeshIO::AddNodeData(
        std::string         aName,
        Plato::ScalarVector aData,
        StrList             aDofNames
    )
    {
        auto tNumDofs = aDofNames.size();
        auto tNumNodes = mMesh.NumNodes();

        if( aData.size() != tNumNodes*tNumDofs )
        {
            ANALYZE_THROWERR("Dimension mismatch!");
        }

        Plato::ScalarMultiVector tData("node data", tNumDofs, tNumNodes);

        auto tStride = aDofNames.size();
        for( decltype(tNumDofs) tIndex=0; tIndex<tNumDofs; tIndex++ )
        {
            auto tScalarVector = Plato::get_vector_component(aData, tIndex, tStride);
            Plato::ScalarVector tSubView = Kokkos::subview(tData, tIndex, Kokkos::ALL());
            Kokkos::deep_copy(tSubView, tScalarVector);
        }

        Variable tAddVar(tData, aDofNames);
        if( mVariablesAreSet )
        {
            setNodeData(aName, tAddVar);
        }
        else
        {
            addNodeData(aName, tAddVar);
        }
    }
    void EngineMeshIO::addNodeData(std::string aName, Variable & aVar) { mNodeVars[aName] = aVar; }
    void EngineMeshIO::setNodeData(std::string aName, Variable & aVar)
    {
        if( mNodeVars.count(aName) == 0 )
        {
            ANALYZE_THROWERR("Attempted to add variable after first write.  EngineMeshIO requires uniform data planes");
        }
        mNodeVars.at(aName) = aVar;
    }

    void EngineMeshIO::AddElementData(
        std::string         aName,
        Plato::ScalarVector aData
    )
    {
        auto tNumElem = aData.extent(0);

        Plato::ScalarMultiVector tData("element data", /*tNumDofs=*/1, tNumElem);
        Plato::ScalarVector tSubView = Kokkos::subview(tData, 0, Kokkos::ALL());
        Kokkos::deep_copy(tSubView, aData);
        AddElementData(aName, tData, {aName});
    }

    void EngineMeshIO::AddElementData(
        std::string              aName,
        Plato::ScalarMultiVector aData
    )
    {
        auto tNumElem = aData.extent(0);
        auto tNumDofs = aData.extent(1);

        Plato::ScalarMultiVector tData("element data", tNumDofs, tNumElem);

        StrList tNames;
        PairList tSuffixes;

        if( tNumDofs == 1 )
        {
            tSuffixes = mScalarSuffixes;
        } else
        if( tNumDofs == mVectorSuffixes.size() )
        {
            tSuffixes = mVectorSuffixes;
        } else
        if( tNumDofs == mSymtensorSuffixes.size() )
        {
            tSuffixes = mSymtensorSuffixes;
        }
        for( const auto& tEntry : tSuffixes )
        {
            const auto& tDofSuffix = tEntry.first;
            const auto& tDofIndex  = tEntry.second;
            std::string tSpace(" ");

            Plato::ScalarVector tSubView = Kokkos::subview(tData, tDofIndex, Kokkos::ALL());
            extractColumn(aData, tDofIndex, tSubView);
            tNames.push_back(aName+tSpace+tDofSuffix);
        }
        AddElementData(aName, tData, tNames);
    }

    void
    EngineMeshIO::extractColumn(
        const Plato::ScalarMultiVector & aFromVector,
              Plato::OrdinalType         aColIndex,
        const Plato::ScalarVector      & aToVector
    )
    {
        auto tNumRows = aFromVector.extent(0);
        if( tNumRows != aToVector.size() )
        {
            ANALYZE_THROWERR("Dimension mismatch");
        }

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            aToVector(aOrdinal) = aFromVector(aOrdinal, aColIndex);
        }, "extract column");
    }

    void EngineMeshIO::AddElementData(
        std::string              aName,
        Plato::ScalarMultiVector aData,
        StrList                  aNames
    )
    {
        if( aData.extent(1) != mMesh.NumElements() )
        {
            ANALYZE_THROWERR("Dimension mismatch!");
        }

        Variable tAddVar(aData, aNames);
        if( mVariablesAreSet )
        {
            setElementData(aName, tAddVar);
        }
        else
        {
            addElementData(aName, tAddVar);
        }
    }
    void EngineMeshIO::addElementData(std::string aName, Variable & aVar) { mElementVars[aName] = aVar; }
    void EngineMeshIO::setElementData(std::string aName, Variable & aVar)
    {
        if( mElementVars.count(aName) == 0 )
        {
            ANALYZE_THROWERR("Attempted to add variable after first write.  EngineMeshIO requires uniform data planes");
        }
        mElementVars.at(aName) = aVar;
    }

    void EngineMeshIO::Write(
        Plato::OrdinalType aStepIndex,
        Plato::Scalar      aTimeValue
    )
    {
        if( aStepIndex != mPlotIndex )
        {
            ANALYZE_THROWERR("EngineMeshIO requires sequential writes.");
        }
        mMeshIO->writeTime(mPlotIndex, aTimeValue);

        if( mVariablesAreSet == false )
        {
            StrList tNodeVarNames;
            for( auto& tNodeVarPair : mNodeVars )
            {
                auto& tNodeVar = tNodeVarPair.second;
                auto& tDofNames = tNodeVar.DofNames;
                for( auto& tDofName : tDofNames )
                {
                    tNodeVar.VarIndices.push_back(tNodeVarNames.size());
                    tNodeVarNames.push_back(tDofName);
                }
            }
            if (tNodeVarNames.size() > 0)
            {
                mMeshIO->initVars(NODE, tNodeVarNames.size(), tNodeVarNames);
            }

            StrList tElementVarNames;
            for( auto& tElementVarPair : mElementVars )
            {
                auto& tElementVar = tElementVarPair.second;
                auto& tDofNames = tElementVar.DofNames;
                for( auto& tDofName : tDofNames )
                {
                    tElementVar.VarIndices.push_back(tElementVarNames.size());
                    tElementVarNames.push_back(tDofName);
                }
            }
            if (tElementVarNames.size() > 0)
            {
                mMeshIO->initVars(ELEM, tElementVarNames.size(), tElementVarNames);
            }

            mVariablesAreSet = true;
        }

        for( auto& tNodeVarPair : mNodeVars )
        {
            Variable& tNodeVar = tNodeVarPair.second;
            if( tNodeVar.isStale )
            {
                std::stringstream tMsg;
                tMsg << "File output (step "<< aStepIndex << ") -- Variable '" 
                     << tNodeVarPair.first << "' not updated. Using previous step.";
                REPORT(tMsg.str());
            }

            auto tHostData = Kokkos::create_mirror_view(tNodeVar.Data);
            Kokkos::deep_copy(tHostData, tNodeVar.Data);

            auto tStride = tNodeVar.DofNames.size();
            for( decltype(tStride) tIndex=0; tIndex<tStride; tIndex++)
            {
                Plato::HostScalarVector tSubView = Kokkos::subview(tHostData, tIndex, Kokkos::ALL());
                mMeshIO->writeNodePlot(tSubView.data(), tNodeVar.VarIndices[tIndex], mPlotIndex);
            }
            tNodeVar.isStale = true;
        }

        for( auto& tElementVarPair : mElementVars )
        {
            Variable& tElementVar = tElementVarPair.second;
            if( tElementVar.isStale )
            {
                std::stringstream tMsg;
                tMsg << "File output (step "<< aStepIndex << ") -- Variable "
                     << tElementVarPair.first << " not updated. using previous step.";
                REPORT(tMsg.str());
            }

            auto tHostData = Kokkos::create_mirror_view(tElementVar.Data);
            Kokkos::deep_copy(tHostData, tElementVar.Data);

            auto tStride = tElementVar.DofNames.size();
            for( decltype(tStride) tIndex=0; tIndex<tStride; tIndex++)
            {
                Plato::HostScalarVector tSubView = Kokkos::subview(tHostData, tIndex, Kokkos::ALL());
                mMeshIO->writeElemPlot(tSubView.data(), tElementVar.VarIndices[tIndex], mPlotIndex);
            }
            tElementVar.isStale = true;
        }

        mVariablesAreSet = true;
        mPlotIndex++;
    }

    Plato::OrdinalType
    EngineMeshIO::NumTimeSteps()
    {
        return mMeshIO->getNumSteps();
    }

    void
    EngineMeshIO::readNodeScalar(
        const std::string        & aVariableName,
              Plato::OrdinalType   aStepIndex,
              Plato::ScalarVector  aNodeScalar
    )
    {
        auto tHostMirror = Kokkos::create_mirror_view(aNodeScalar);
        auto tSuccess = mMeshIO->readNodePlot(tHostMirror.data(), aVariableName, aStepIndex);
        if( !tSuccess )
        {
            ANALYZE_PRINTERR("Reading from mesh failed.");
        }
        Kokkos::deep_copy(aNodeScalar, tHostMirror);
    }

    Plato::ScalarVector EngineMeshIO::ReadNodeData(
        const std::string        & aVariableName,
              Plato::OrdinalType   aStepIndex
    )
    {
        auto tNumNodes = mMesh.NumNodes();

        if( mVectorNodeVars.count(aVariableName) )
        {
            auto tVectorNodeVar = mVectorNodeVars.at(aVariableName);
            auto tNumTerms = tVectorNodeVar.size();

            Plato::ScalarVector tReturnData("vector data", tNumTerms*tNumNodes);

            Plato::ScalarMultiVector tDataFromFile("data from file", tNumTerms, tNumNodes);
            for( decltype(tNumTerms) iTerm=0; iTerm<tNumTerms; iTerm++ )
            {
                Plato::ScalarVector tSubView = Kokkos::subview(tDataFromFile, iTerm, Kokkos::ALL());
                readNodeScalar(tVectorNodeVar[iTerm], aStepIndex, tSubView);
                set_vector_component(tReturnData, tSubView, iTerm, tNumTerms);
            }
            return tReturnData;
        }
        else
        {
            Plato::ScalarVector tDataFromFile("data from file", tNumNodes);
            readNodeScalar(aVariableName, aStepIndex, tDataFromFile);
            return tDataFromFile;
        }
    }
}
