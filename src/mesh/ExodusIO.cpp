#include "ExodusIO.hpp"
#include "PlatoUtilities.hpp"

#include <exodusII.h>

namespace Plato
{
    using Int = ExodusIO::Int;
    using Real = ExodusIO::Real;

    std::vector<std::vector<Int>>
    ExodusIO::getFaceGraph(Int aBlockIndex) const
    {
        return getFaceGraph(mElementBlocks[aBlockIndex].elementType);
    }

    std::vector<std::vector<Int>>
    ExodusIO::getFaceGraph(std::string aElemType) const
    {
        // See the ExodusII reference manual for face node numbering
        
        auto tElemType = Plato::tolower(aElemType);

        if (tElemType == "beam" || tElemType == "bar2")
        {
            return {{0},{1}};
        }
        else
        if (tElemType == "tri3")
        {
            return {{0,1},{1,2},{2,0}};
        } 
        else
        if (tElemType == "tri" || tElemType == "tri3")
        {
            return {{0,1},{1,2},{2,0}};
        } 
        else
        if (tElemType == "quad" || tElemType == "quad4")
        {
            return {{0,1},{1,2},{2,3},{3,0}};
        } 
        else
        if (tElemType == "quad8" || tElemType == "quad9")
        {
            return {{0,1,4},{1,2,5},{2,3,6},{3,0,7}};
        } 
        else
        if (tElemType == "tetra" || tElemType == "tetra4" || tElemType == "tet4")
        {
            return {{0,1,3},{1,2,3},{2,0,3},{0,2,1}};
        } 
        else
        if (tElemType == "tetra10" || tElemType == "tet10")
        {
            return {{0,1,3,4,8,7},{1,2,3,5,9,8},{2,0,3,6,7,9},{0,2,1,6,5,4}};
        } 
        else
        if (tElemType == "hex" || tElemType == "hex8")
        {
            return {{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7},{3,2,1,0},{4,5,6,7}};
        } 
        else
        if (tElemType == "hex20" || tElemType == "hex21")
        {
            return {{0,1,5,4, 8,13,16,12},
                    {1,2,6,5, 9,14,17,13},
                    {2,3,7,6,10,15,18,14},
                    {3,0,4,7,11,12,19,15},
                    {3,2,1,0,10, 9, 8,11},
                    {4,5,6,7,16,17,18,19}};
        } 
        else
        if (tElemType == "hex27")
        {
            return {{0,1,5,4, 8,13,16,12,25},
                    {1,2,6,5, 9,14,17,13,24},
                    {2,3,7,6,10,15,18,14,26},
                    {3,0,4,7,11,12,19,15,23},
                    {3,2,1,0,10, 9, 8,11,21},
                    {4,5,6,7,16,17,18,19,22}};
        } 
        else
        {
            std::stringstream msg;
            msg << "Unable to read exodus file. Unsupported element type: " << aElemType;
            ANALYZE_THROWERR(msg.str());
        }
    }


    void
    ExodusIO::readMesh(const std::string & aFileName, bool aIgnoreNodeMap, bool aIgnoreElemMap)
    {
        mIgnoreNodeMap = aIgnoreNodeMap;
        mIgnoreElemMap = aIgnoreElemMap;

        openMesh(aFileName, "read");
        readData();
        closeMesh();
    }

    void
    ExodusIO::closeMesh()
    {
        if( mFileID != -1 )
        {
            Int tErrorStatus = ex_close(mFileID);
            if(tErrorStatus) { ANALYZE_THROWERR("Error closing exodus file. ex_close() failed."); }

            mFileID = -1;
        }
    }

    void
    ExodusIO::readData()
    {
        Int tErrorStatus;

        // get numbers of local entities
        char tTitle[256];
        Int tNumBlocks, tNumNodeSets, tNumSideSets;
        tErrorStatus = ex_get_init(mFileID, tTitle, &mNumDims, &mNumNodes, &mNumElems, &tNumBlocks, &tNumNodeSets, &tNumSideSets);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_init() failed."); }

        mNodeGlobalIds.resize(mNumNodes);
        readNodeIds();

        mElemGlobalIds.resize(mNumElems);
        readElemIds();

        mNodeSets.resize(tNumNodeSets);
        readNodeSets();

        mSideSets.resize(tNumSideSets);
        readSideSets();

        mElementBlocks.resize(tNumBlocks);
        readElementBlocks();

        mCoords.resize(mNumDims, std::vector<Real>(mNumNodes));
        readCoords();
    }

    void
    ExodusIO::readCoords()
    {
        auto x = (mNumDims > 0) ? mCoords[0].data() : nullptr;
        auto y = (mNumDims > 1) ? mCoords[1].data() : nullptr;
        auto z = (mNumDims > 2) ? mCoords[2].data() : nullptr;

        Int tErrorStatus = ex_get_coord(mFileID, x, y, z);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_coord() failed."); }
    }

    void
    ExodusIO::readElemIds()
    {
        if(mIgnoreElemMap)
        {
            for( Int iElem=0; iElem<mNumElems; iElem++ ){
                mElemGlobalIds[iElem] = iElem+1;
            }
        } else {
            Int tMapCount = ex_inquire_int(mFileID, EX_INQ_ELEM_MAP);
            if(tMapCount == 1)
            {
                ex_get_num_map(mFileID, EX_ELEM_MAP, 1, mElemGlobalIds.data());
            }
            else
            {
                ex_get_id_map(mFileID, EX_ELEM_MAP, mElemGlobalIds.data());
            }
        }
    }

    void
    ExodusIO::readElementBlocks()
    {
        Int tErrorStatus;

        Int tNumElementBlocks = mElementBlocks.size();

        if(tNumElementBlocks)
        {
            std::vector<Int> tBlockIds(tNumElementBlocks);
            tErrorStatus = ex_get_ids(mFileID, EX_ELEM_BLOCK, tBlockIds.data());
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_ids() failed."); }

            char** tBlockNames = new char*[tNumElementBlocks];
            for(Int i=0; i<tNumElementBlocks; i++)
               tBlockNames[i] = new char[256];

            tErrorStatus = ex_get_names(mFileID, EX_ELEM_BLOCK, tBlockNames);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_names() failed."); }

            for(Int i=0; i<tNumElementBlocks; i++)
            {
                char tElemType[16];
                Int tNumElemsInBlock, tNumNodesPerElem, tNumEdgesPerElem, tNumFacesPerElem, tNumAttr;
                tErrorStatus = ex_get_block( mFileID, EX_ELEM_BLOCK, tBlockIds[i], tElemType,
                                             &tNumElemsInBlock, &tNumNodesPerElem, &tNumEdgesPerElem,
                                             &tNumFacesPerElem, &tNumAttr);
                if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_block() failed."); }

                mElementBlocks[i].ID = tBlockIds[i];
                mElementBlocks[i].name = std::string(tBlockNames[i]);
                mElementBlocks[i].numNPE = tNumNodesPerElem;
                mElementBlocks[i].numElems = tNumElemsInBlock;
                mElementBlocks[i].elementType = std::string(tElemType);
                mElementBlocks[i].faceGraph = getFaceGraph(tElemType);

                auto tNumConn = tNumNodesPerElem*tNumElemsInBlock;
                mElementBlocks[i].connectivity.resize(tNumConn);
                auto tConnect = mElementBlocks[i].connectivity.data();
                tErrorStatus = ex_get_conn( mFileID, EX_ELEM_BLOCK, tBlockIds[i], tConnect, 0, 0);
                if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_conn() failed."); }

                // from one-based to zero-based indices
                for(Int iConn=0; iConn<tNumConn; iConn++)
                  tConnect[iConn] = tConnect[iConn]-1;
            }

            for(Int i=0; i<tNumElementBlocks; i++)
               delete [] tBlockNames[i];
            delete [] tBlockNames;
        }
    }

    void
    ExodusIO::readSideSets()
    {
        Int tErrorStatus;

        Int tNumSideSets = mSideSets.size();

        if(tNumSideSets)
        {
            std::vector<Int> tSideSetIds(tNumSideSets);
            tErrorStatus = ex_get_ids(mFileID, EX_SIDE_SET, tSideSetIds.data());
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_ids() failed."); }

            char** tSideSetNames = new char*[tNumSideSets];
            for(Int i=0; i<tNumSideSets; i++)
               tSideSetNames[i] = new char[256];

            tErrorStatus = ex_get_names(mFileID, EX_SIDE_SET, tSideSetNames);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_names() failed."); }

            for(Int i=0; i<tNumSideSets; i++)
            {
                Int tNumSidesInSet, tNumDistsInSet;
                tErrorStatus = ex_get_set_param(mFileID, EX_SIDE_SET, tSideSetIds[i], &tNumSidesInSet, &tNumDistsInSet);
                if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_set_param() failed."); }

                Int tNumNodesInSet;
                tErrorStatus = ex_get_side_set_node_list_len(mFileID, tSideSetIds[i], &tNumNodesInSet);
                if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_side_set_node_list_len() failed."); }

                mSideSets[i].ID = tSideSetIds[i];
                mSideSets[i].elems.resize(tNumSidesInSet);
                mSideSets[i].sides.resize(tNumSidesInSet);
                mSideSets[i].nodes.resize(tNumNodesInSet);
                mSideSets[i].name = std::string(tSideSetNames[i]);


                if(tNumSidesInSet)
                {
                    auto tElems = mSideSets[i].elems.data();
                    auto tSides = mSideSets[i].sides.data();
                    tErrorStatus = ex_get_set(mFileID, EX_SIDE_SET, tSideSetIds[i], tElems, tSides);
                    if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_set() failed."); }

                    auto tNodes = mSideSets[i].nodes.data();
                    std::vector<Int> tNodesPerFace(tNumSidesInSet);
                    tErrorStatus = ex_get_side_set_node_list(mFileID, tSideSetIds[i], tNodesPerFace.data(), tNodes);
                    if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_side_set_node_list() failed."); }

                    mSideSets[i].numNPF = tNodesPerFace[0];

                    // switch from one-based to zero-based indexing
                    for(Int iside=0; iside<tNumSidesInSet; iside++)
                    {
                        tSides[iside] -= 1;
                        tElems[iside] -= 1;
                        for(Int inode=0; inode<mSideSets[i].numNPF; inode++)
                            tNodes[iside*mSideSets[i].numNPF+inode] -= 1;
                    }
                }
            }
            for(Int i=0; i<tNumSideSets; i++)
               delete [] tSideSetNames[i];
            delete [] tSideSetNames;
        }
    }

    void
    ExodusIO::readNodeSets()
    {
        Int tNumNodeSets = mNodeSets.size();

        if(tNumNodeSets)
        {
            Int tErrorStatus;

            std::vector<Int> tNodeSetIds(tNumNodeSets);
            tErrorStatus = ex_get_ids(mFileID, EX_NODE_SET, tNodeSetIds.data());
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_ids() failed."); }

            char** tNodeSetNames = new char*[tNumNodeSets];
            for(Int i=0; i<tNumNodeSets; i++)
               tNodeSetNames[i] = new char[256];

            tErrorStatus = ex_get_names(mFileID, EX_NODE_SET, tNodeSetNames);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_names() failed."); }

            for(Int i=0; i<tNumNodeSets; i++)
            {
                Int tNumNodesInSet, tNumDistsInSet;
                tErrorStatus = ex_get_set_param(mFileID, EX_NODE_SET, tNodeSetIds[i], &tNumNodesInSet, &tNumDistsInSet);
                if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_set_param() failed."); }

                mNodeSets[i].ID = tNodeSetIds[i];
                mNodeSets[i].nodes.resize(tNumNodesInSet);
                mNodeSets[i].name = std::string(tNodeSetNames[i]);

                if(tNumNodesInSet)
                {
                    auto tNodes = mNodeSets[i].nodes.data();
                    tErrorStatus = ex_get_set(mFileID, EX_NODE_SET, tNodeSetIds[i], tNodes, NULL);
                    if(tErrorStatus) { ANALYZE_THROWERR("Unable to read exodus file. ex_get_set() failed."); }
                    for(Int j=0; j<tNumNodesInSet; j++)
                    {
                        tNodes[j] = tNodes[j] - 1;
                    }
                }
            }
            for(Int i=0; i<tNumNodeSets; i++)
               delete [] tNodeSetNames[i];
            delete [] tNodeSetNames;
        }
    }

    void
    ExodusIO::readNodeIds()
    {
        if(mIgnoreNodeMap){
            for( Int iNode=0; iNode<mNumNodes; iNode++ ){
                mNodeGlobalIds[iNode] = iNode+1;
            }
        } else {
            Int tMapCount = ex_inquire_int(mFileID, EX_INQ_NODE_MAP);
            if(tMapCount == 1)
            {
                ex_get_num_map(mFileID, EX_NODE_MAP, 1, mNodeGlobalIds.data());
            }
            else
            {
                ex_get_id_map(mFileID, EX_NODE_MAP, mNodeGlobalIds.data());
            }
        }
    }

    void ExodusIO::checkAddExtension(std::string & aName, std::string aExt)
    {
        auto found = aName.find(aExt);
        if( found != (aName.size() - aExt.size()) )
        {
            aName += aExt;
        }
    }

    void
    ExodusIO::openMesh(std::string aFileName, const std::string & aMode)
    {

        closeMesh();

        checkAddExtension(aFileName, ".exo");

        Int CPU_word_size = 8;
        Int IO_word_size = 8;
        float version = 0.0;

        auto tMode = Plato::tolower(aMode);
        if(tMode == "read" || tMode == "r")
        {
            mFileID = ex_open(aFileName.c_str(), EX_READ, &CPU_word_size, &IO_word_size, &version);
        }
        else
        if(tMode == "write" || tMode == "w")
        {
            mFileID = ex_create(aFileName.c_str(), EX_CLOBBER, &CPU_word_size, &IO_word_size);
        }
        else
        if(tMode == "append" || tMode == "a")
        {
            mFileID = ex_open(aFileName.c_str(), EX_WRITE, &CPU_word_size, &IO_word_size, &version);
        }
        else
        {
            std::stringstream msg;
            msg << "Unknown access mode requested: " << aMode << ". Opening exodus file failed.";
            ANALYZE_THROWERR(msg.str());
        }

        if(mFileID <= 0)
        {
            std::stringstream msg;
            msg << "Unable to read exodus file: " << aFileName;
            ANALYZE_THROWERR(msg.str());
        }

        if(tMode == "write"  || tMode == "w" ||
           tMode == "append" || tMode == "a"  )
        {
            writeData();
        }
    }

    void
    ExodusIO::writeData()
    {
        Int tErrorStatus;

        char tTitle[24] = "Analyze output";
        tErrorStatus = ex_put_init(mFileID, tTitle, mNumDims, mNumNodes, mNumElems, mElementBlocks.size(), mNodeSets.size(), mSideSets.size());
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_init() failed."); }

        tErrorStatus = ex_put_id_map(mFileID, EX_NODE_MAP, mNodeGlobalIds.data());
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_id_map() failed."); }

        tErrorStatus = ex_put_id_map(mFileID, EX_ELEM_MAP, mElemGlobalIds.data());
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_id_map() failed."); }

        for(auto& tNodeSet : mNodeSets )
        {
            tErrorStatus = ex_put_set_param(mFileID, EX_NODE_SET, tNodeSet.ID, tNodeSet.nodes.size(), /*num_dist_in_set=*/ 0);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_set_param() failed."); }

            std::vector<Int> tNodesOneBased(tNodeSet.nodes);
            for( auto & tNodeID : tNodesOneBased ) tNodeID++;
            tErrorStatus = ex_put_set(mFileID, EX_NODE_SET, tNodeSet.ID, tNodesOneBased.data(), NULL);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_set() failed."); }
        }

        for(auto& tSideSet : mSideSets )
        {
            tErrorStatus = ex_put_set_param(mFileID, EX_SIDE_SET, tSideSet.ID, tSideSet.elems.size(), /*num_dist_in_set=*/ 0);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_set_param() failed."); }

            std::vector<Int> tElemsOneBased(tSideSet.elems);
            std::vector<Int> tSidesOneBased(tSideSet.sides);
            for( auto & tElemID : tElemsOneBased ) tElemID++;
            for( auto & tFaceID : tSidesOneBased ) tFaceID++;
            tErrorStatus = ex_put_set(mFileID, EX_SIDE_SET, tSideSet.ID, tElemsOneBased.data(), tSidesOneBased.data());
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_set() failed."); }
        }

        for(auto& tBlock : mElementBlocks )
        {
            tErrorStatus = ex_put_block(mFileID, EX_ELEM_BLOCK, tBlock.ID, tBlock.elementType.c_str(), tBlock.numElems, tBlock.numNPE, 0, 0, 0 );
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_block() failed."); }

            std::vector<Int> tConnectOneBased(tBlock.connectivity);
            for( auto & tConnID : tConnectOneBased ) tConnID++;
            tErrorStatus = ex_put_conn(mFileID, EX_ELEM_BLOCK, tBlock.ID, tConnectOneBased.data(), 0, 0);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_conn() failed."); }
        }

        tErrorStatus = ex_put_map_param(mFileID, 1, 1);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_map_param() failed."); }

        
        auto x = (mNumDims > 0) ? mCoords[0].data() : nullptr;
        auto y = (mNumDims > 1) ? mCoords[1].data() : nullptr;
        auto z = (mNumDims > 2) ? mCoords[2].data() : nullptr;
        tErrorStatus = ex_put_coord(mFileID, x, y, z);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write exodus file. ex_put_coords() failed."); }

    }

    Int ExodusIO::getNumElemBlks() const { return mElementBlocks.size(); }
    Int ExodusIO::getNumElemInBlk(Int blk) const { return mElementBlocks[blk].numElems; }
    Int ExodusIO::getNnpeInBlk(Int blk) const { return mElementBlocks[blk].numNPE; }
    Int ExodusIO::getDimensions() const { return mNumDims; }

    const std::vector<Int> &
    ExodusIO::getElemToNodeConnInBlk(Int blk) const { return mElementBlocks[blk].connectivity; }

    const std::vector<std::vector<Real>> &
    ExodusIO::getCoords() const { return mCoords; }

    std::string
    ExodusIO::getElemTypeInBlk(Int blk) const { return mElementBlocks[blk].elementType; }
    std::string
    ExodusIO::getBlockName(Int blk) const { return mElementBlocks[blk].name; }

    Int ExodusIO::getNumNodeSets() const { return mNodeSets.size(); }
    Int ExodusIO::getNodeSetLength(Int i) const { return mNodeSets[i].nodes.size(); }
    std::string ExodusIO::getNodeSetName(Int i) const { return mNodeSets[i].name; }
    const std::vector<Int> &
    ExodusIO::getNodeSetNodes(Int i) const { return mNodeSets[i].nodes; }

    Int ExodusIO::getNumSideSets() const { return mSideSets.size(); }
    Int ExodusIO::getSideSetLength(Int i) const { return mSideSets[i].sides.size(); }
    Int ExodusIO::getSideSetNodesPerFace(Int i) const { return mSideSets[i].numNPF; }
    std::string ExodusIO::getSideSetName(Int i) const { return mSideSets[i].name; }

    const std::vector<Int> &
    ExodusIO::getSideSetFaces(Int i) const { return mSideSets[i].sides; }
    const std::vector<Int> &
    ExodusIO::getSideSetElems(Int i) const { return mSideSets[i].elems; }
    const std::vector<Int> &
    ExodusIO::getSideSetNodes(Int i) const { return mSideSets[i].nodes; }

    /******************************************************************************//**
    * \brief Get nodal variable names from the exodus database
    **********************************************************************************/
    std::vector<std::string>
    ExodusIO::getNodeVarNames() const
    {
        std::vector<std::string> tReturnNames;

        Int tNumNodeVars;
        ex_get_variable_param(mFileID, EX_NODAL, &tNumNodeVars);

        char** tNames = new char*[tNumNodeVars];
        for(Int i=0; i<tNumNodeVars; i++)
            tNames[i] = new char[MAX_STR_LENGTH+1];

        ex_get_variable_names(mFileID, EX_NODAL, tNumNodeVars, tNames);

        for(Int i=0; i<tNumNodeVars; i++)
        {
            std::string tVarName(tNames[i]);
            tReturnNames.push_back(tVarName);
            delete [] tNames[i];
        }
        delete [] tNames;

        return tReturnNames;
    }

    /******************************************************************************//**
    * \brief Begin writing new time plane.
    * \param [in] aTimeStep Zero-based index into time steps array
    * \param [in] aTimeValue Time value.  Value must be greater than the previous time value.
    **********************************************************************************/
    void
    ExodusIO::writeTime(Int aTimeStep, Real aTimeValue) const
    {
        auto tOneBasedTimeStep = aTimeStep + 1;
        Int tErrorStatus = ex_put_time(mFileID, tOneBasedTimeStep, &aTimeValue);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_put_time() failed."); }
    }

    void
    ExodusIO::initVars(std::string aCentering,
                       Int aNumVars,
                       std::vector<std::string> aNames ) const
    {
        Int tErrorStatus;

        char** tNames = new char* [aNumVars];
        for(Int i=0; i<aNumVars; i++)
        {
            tNames[i] = (char*) aNames[i].c_str();
        }

        if( aCentering == "node" )
        {
            tErrorStatus = ex_put_variable_param(mFileID, EX_NODAL, aNumVars);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_put_variable_param() failed."); }

            tErrorStatus = ex_put_variable_names(mFileID, EX_NODAL, aNumVars, tNames);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_put_variable_names() failed."); }
        }
        else
        if( aCentering == "element" )
        {
            tErrorStatus = ex_put_variable_param(mFileID, EX_ELEM_BLOCK, aNumVars);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_put_variable_param() failed."); }

            tErrorStatus = ex_put_variable_names(mFileID, EX_ELEM_BLOCK, aNumVars, tNames);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_put_variable_names() failed."); }
        }
        else
        {
            std::stringstream msg;
            msg << "Unable to write exodus data. Unsupported centering: " << aCentering;
            ANALYZE_THROWERR(msg.str());
        }
        delete [] tNames;
    }

    /******************************************************************************//**
    * \brief Write node plot to exodus mesh
    * \param [in] aData pointer to node data of length numNodes
    * \param [in] aVariableIndex zero-based index into nodal variables array
    * \param [in] aStepIndex zero-based index into time step array
    **********************************************************************************/
    void
    ExodusIO::writeNodePlot(Real* aData, Int aVariableIndex, Int aStepIndex) const
    {
        Int tErrorStatus;

        // index arguments to this function are zero-based.  The exodus API is one-based:
        auto tOneBasedVariableIndex = aVariableIndex+1;
        auto tOneBasedStepIndex = aStepIndex+1;

        tErrorStatus = ex_put_var(mFileID, tOneBasedStepIndex, EX_NODAL, tOneBasedVariableIndex, /*obj_id=*/1, mNumNodes, aData);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_put_var() failed."); }

        tErrorStatus = ex_update(mFileID);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_update() failed."); }
    }

    /******************************************************************************//**
    * \brief Write element plot to exodus mesh
    * \param [in] aData pointer to element data of length numElements
    * \param [in] aVariableIndex zero-based index into element variables array
    * \param [in] aStepIndex zero-based index into time step array
    **********************************************************************************/
    void
    ExodusIO::writeElemPlot(Real* aData, Int aVariableIndex, Int aStepIndex) const
    {
        Int tErrorStatus;

        // index arguments to this function are zero-based.  The exodus API is one-based:
        Int tOneBasedVariableIndex = aVariableIndex+1;
        Int tOneBasedStepIndex = aStepIndex+1;

        Real *tOffsetData = NULL;
        Int tElemCount = 0;
        for(auto& tBlock : mElementBlocks)
        {
            if(tBlock.numElems == 0) continue;
            tOffsetData = &aData[tElemCount];
            tErrorStatus = ex_put_var(mFileID, tOneBasedStepIndex, EX_ELEM_BLOCK, tOneBasedVariableIndex, tBlock.ID, tBlock.numElems, tOffsetData);
            if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_put_var() failed."); }
            tElemCount += tBlock.numElems; //assumes all blocks have same data
        }
        tErrorStatus = ex_update(mFileID);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to write data. ex_update() failed."); }
    }

    /******************************************************************************//**
    * \brief Get the number of time steps stored in the exodus database
    **********************************************************************************/
    Int
    ExodusIO::getNumSteps() const
    {
         return ex_inquire_int(mFileID, EX_INQ_TIME);
    }

    /******************************************************************************//**
    * \brief Read node plot from exodus mesh
    * \param [out] aData Pointer to return data
    * \param [in] aVariableName Name of nodal variable to read
    * \param [in] aStepIndex zero-based index into time step array
    **********************************************************************************/
    void
    ExodusIO::readNodePlot(double* aData, std::string aVariableName, Int aStepIndex) const
    {
        Int tErrorStatus;

        Int tNumTimeSteps;
        float tDummyFloat;
        char tDummyChar;
        tErrorStatus = ex_inquire(mFileID, EX_INQ_TIME, &tNumTimeSteps, &tDummyFloat, &tDummyChar);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to read data. ex_inquire() failed."); }

        Int tReadTimeStep = -1;
        Int tOneBasedStepIndex = aStepIndex+1;
        if( tOneBasedStepIndex == 0 ) {
            tReadTimeStep = tNumTimeSteps;
        } else {
            if( tOneBasedStepIndex > tNumTimeSteps || tOneBasedStepIndex < 1 )
            {
                ANALYZE_THROWERR("Fatal Error: Requested time step doesn't exist");
            }
            tReadTimeStep = tOneBasedStepIndex;
        }

        Int tNumNodeVars;
        tErrorStatus = ex_get_variable_param(mFileID, EX_NODAL, &tNumNodeVars);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to read data. ex_get_variable_param() failed."); }

        char** tNames = new char*[tNumNodeVars];
        for(Int i=0; i<tNumNodeVars; i++)
             tNames[i] = new char[MAX_STR_LENGTH+1];
        tErrorStatus = ex_get_variable_names(mFileID, EX_NODAL, tNumNodeVars, tNames);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to read data. ex_get_variable_names() failed."); }

        Int tOneBasedVarIndex=-1;
        for(Int i=0; i<tNumNodeVars; i++)
        {
            if(!strncasecmp(tNames[i], aVariableName.c_str(), aVariableName.length()))
            {
                tOneBasedVarIndex=i+1;
                break;
            }
        }

        if (tOneBasedVarIndex < 1)
        {
            std::stringstream tMessage;
            tMessage << "Fatal Error: Requested variable that doesn't exist" << std::endl;
            tMessage << "Requested: " << aVariableName << std::endl;
            tMessage << "Available: " << std::endl;
            for(Int iName=0; iName<tNumNodeVars; iName++)
            {
                tMessage << "  " << tNames[iName] << std::endl;
            }
            ANALYZE_THROWERR(tMessage.str());
        }

        for(Int i=0; i<tNumNodeVars; i++)
           delete [] tNames[i];
        delete [] tNames;

        tErrorStatus = ex_get_var(mFileID, tReadTimeStep, EX_NODAL, tOneBasedVarIndex, 1, mNumNodes, aData);
        if(tErrorStatus) { ANALYZE_THROWERR("Unable to read data. ex_get_var() failed."); }
    }

} // end namespace Plato
