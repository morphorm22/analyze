#include "InputDataUtils.hpp"

namespace Plato
{
  void 
  readCSVData(
    std::string            const & aFileName,
    Teuchos::ParameterList       & aInputs,
    Plato::DataMap               & aDataMap,
    Plato::Mesh                    aMesh
  )
  {
    CSVReader tReader(aInputs, aDataMap, aMesh);

    std::string tLine;
    std::ifstream tInfile;
    tInfile.open (aFileName);

    if(tInfile.is_open() == false)
    {
      std::stringstream ss;
      ss << "Failed to open file '" << aFileName << "'.";
      ANALYZE_THROWERR(ss.str());
    }
 
    while(!tInfile.eof())
    {
      std::getline(tInfile, tLine);
      tReader.readLine(tLine);
    }
    tReader.close();
    tInfile.close();
  }

  void
  verifyOrthogonality(
    Plato::ScalarVector a1X,
    Plato::ScalarVector a1Y,
    Plato::ScalarVector a1Z,
    Plato::ScalarVector a2X,
    Plato::ScalarVector a2Y,
    Plato::ScalarVector a2Z,
    Plato::Scalar       aTol
  )
  {
    Plato::OrdinalType tNumErr = 0;
    auto tLength = a1X.extent(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tLength),
    KOKKOS_LAMBDA(const Plato::OrdinalType & tDof, Plato::OrdinalType & tSum)
    {
      if(a1X(tDof)*a2X(tDof)+a1Y(tDof)*a2Y(tDof)+a1Z(tDof)*a2Z(tDof) > aTol)
      {
        tSum += 1;
      }
    }, tNumErr);
    if(tNumErr)
    {
      ANALYZE_THROWERR("Found basis vectors that are not orthogonal.");
    }
  }

  void
  verifyOrthogonality(
    Plato::ScalarVector a1X,
    Plato::ScalarVector a1Y,
    Plato::ScalarVector a2X,
    Plato::ScalarVector a2Y,
    Plato::Scalar       aTol
  )
  {
    Plato::OrdinalType tNumErr = 0;
    auto tLength = a1X.extent(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tLength),
    KOKKOS_LAMBDA(const Plato::OrdinalType & tDof, Plato::OrdinalType & tSum)
    {
      if(a1X(tDof)*a2X(tDof)+a1Y(tDof)*a2Y(tDof) > aTol)
      {
        tSum += 1;
      }
    }, tNumErr);
    if(tNumErr)
    {
      ANALYZE_THROWERR("Found basis vectors that are not orthogonal.");
    }
  }

  void
  normalizeVector(
    Plato::ScalarVector aX,
    Plato::ScalarVector aY,
    Plato::ScalarVector aZ
  )
  {
    Plato::OrdinalType tNumErr = 0;
    auto tLength = aX.extent(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tLength),
    KOKKOS_LAMBDA(const Plato::OrdinalType & tDof, Plato::OrdinalType & tSum)
    {
      auto tX = aX(tDof);
      auto tY = aY(tDof);
      auto tZ = aZ(tDof);
      auto tMag = tX*tX+tY*tY+tZ*tZ;

      if( tMag == 0 )
      { 
        tSum += 1;
      }
  
      tMag = sqrt(tMag);

      decltype(tX) tOne(1.0);
      if(fabs(tOne - tMag) > DBL_EPSILON)
      {
        aX(tDof) /= tMag;
        aY(tDof) /= tMag;
        aZ(tDof) /= tMag;
      }
    }, tNumErr);
    if(tNumErr)
    {
      ANALYZE_THROWERR("Found basis vector of length zero.");
    }
  }

  void
  normalizeVector(
    Plato::ScalarVector aX,
    Plato::ScalarVector aY
  )
  {
    Plato::OrdinalType tNumErr = 0;
    auto tLength = aX.extent(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tLength),
    KOKKOS_LAMBDA(const Plato::OrdinalType & tDof, Plato::OrdinalType & tSum)
    {
      auto tX = aX(tDof);
      auto tY = aY(tDof);
      auto tMag = tX*tX+tY*tY;

      if( tMag == 0 )
      { 
        tSum += 1;
      }
  
      tMag = sqrt(tMag);

      decltype(tX) tOne(1.0);
      if(fabs(tOne - tMag) > DBL_EPSILON)
      {
        aX(tDof) /= tMag;
        aY(tDof) /= tMag;
      }
    }, tNumErr);
    if(tNumErr)
    {
      ANALYZE_THROWERR("Attempted to normalize a vector of length zero.");
    }
  }

  Plato::ScalarArray3D
  createOrthonormalTensor2D(
    Teuchos::ParameterList & tTransformParams,
    Plato::DataMap         & aDataMap
  )
  {
    auto tXNames = Plato::ParseTools::getParam<Teuchos::Array<std::string>>(tTransformParams, "X");
    Plato::ParseTools::verifyVectorLength(tXNames, 2);
    auto tX0 = aDataMap.scalarVectors[tXNames[0]];
    auto tX1 = aDataMap.scalarVectors[tXNames[1]];
    Plato::normalizeVector(tX0, tX1);

    Plato::ScalarVector tY0, tY1;
    if(tTransformParams.isType<Teuchos::Array<std::string>>("Y"))
    {
      auto tYNames = Plato::ParseTools::getParam<Teuchos::Array<std::string>>(tTransformParams, "Y");
      Plato::ParseTools::verifyVectorLength(tYNames, 2);
      tY0 = aDataMap.scalarVectors[tYNames[0]];
      tY1 = aDataMap.scalarVectors[tYNames[1]];
      Plato::verifyOrthogonality(tX0, tX1, tY0, tY1, /*tolerance=*/ 1e-6);
    }
    else
    {
      auto tLength = tX0.extent(0);
      tY0 = Plato::ScalarVector("Y0", tLength);
      tY1 = Plato::ScalarVector("Y1", tLength);
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength),
      KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
      {
        tY0(tDof) =-tX1(tDof);
        tY1(tDof) = tX0(tDof);
      }, "perpendicular");
    }

    auto tLength = tX0.extent(0);
    Plato::ScalarArray3D tTensor("orthonormal tensor", tLength, 2, 2);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength),
    KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
    {
      tTensor(tDof, 0, 0) = tX0(tDof);
      tTensor(tDof, 1, 0) = tX1(tDof);
      tTensor(tDof, 0, 1) = tY0(tDof);
      tTensor(tDof, 1, 1) = tY1(tDof);
    }, "create tensor");

    return tTensor;
  }

  Plato::ScalarArray3D
  createOrthonormalTensor3D(
    Teuchos::ParameterList & tTransformParams,
    Plato::DataMap         & aDataMap
  )
  {
    auto tXNames = Plato::ParseTools::getParam<Teuchos::Array<std::string>>(tTransformParams, "X");
    Plato::ParseTools::verifyVectorLength(tXNames, 3);
    auto tX0 = aDataMap.scalarVectors[tXNames[0]];
    auto tX1 = aDataMap.scalarVectors[tXNames[1]];
    auto tX2 = aDataMap.scalarVectors[tXNames[2]];
    Plato::normalizeVector(tX0, tX1, tX2);

    auto tYNames = Plato::ParseTools::getParam<Teuchos::Array<std::string>>(tTransformParams, "Y");
    Plato::ParseTools::verifyVectorLength(tYNames, 3);
    auto tY0 = aDataMap.scalarVectors[tYNames[0]];
    auto tY1 = aDataMap.scalarVectors[tYNames[1]];
    auto tY2 = aDataMap.scalarVectors[tYNames[2]];
    Plato::normalizeVector(tY0, tY1, tY2);

    Plato::verifyOrthogonality(tX0, tX1, tX2, tY0, tY1, tY2, /*tolerance=*/ 1e-6);

    Plato::ScalarVector tZ0, tZ1, tZ2;
    if(tTransformParams.isType<Teuchos::Array<std::string>>("Z"))
    {
      auto tZNames = Plato::ParseTools::getParam<Teuchos::Array<std::string>>(tTransformParams, "Z");
      Plato::ParseTools::verifyVectorLength(tZNames, 3);
      tZ0 = aDataMap.scalarVectors[tZNames[0]];
      tZ1 = aDataMap.scalarVectors[tZNames[1]];
      tZ2 = aDataMap.scalarVectors[tZNames[2]];
      Plato::verifyOrthogonality(tX0, tX1, tX2, tZ0, tZ1, tZ2, /*tolerance=*/ 1e-6);
      Plato::verifyOrthogonality(tY0, tY1, tY2, tZ0, tZ1, tZ2, /*tolerance=*/ 1e-6);
    }
    else
    {
      auto tLength = tX0.extent(0);
      tZ0 = Plato::ScalarVector("Z0", tLength);
      tZ1 = Plato::ScalarVector("Z1", tLength);
      tZ2 = Plato::ScalarVector("Z2", tLength);
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength),
      KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
      {
        tZ0(tDof) = tX1(tDof)*tY2(tDof) - tX2(tDof)*tY1(tDof);
        tZ1(tDof) = tX2(tDof)*tY0(tDof) - tX0(tDof)*tY2(tDof);
        tZ2(tDof) = tX0(tDof)*tY1(tDof) - tX1(tDof)*tY0(tDof);
      }, "cross product");
    }

    auto tLength = tX0.extent(0);
    Plato::ScalarArray3D tTensor("orthonormal tensor", tLength, 3, 3);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength),
    KOKKOS_LAMBDA(const Plato::OrdinalType & tDof)
    {
      tTensor(tDof, 0, 0) = tX0(tDof);
      tTensor(tDof, 1, 0) = tX1(tDof);
      tTensor(tDof, 2, 0) = tX2(tDof);
      tTensor(tDof, 0, 1) = tY0(tDof);
      tTensor(tDof, 1, 1) = tY1(tDof);
      tTensor(tDof, 2, 1) = tY2(tDof);
      tTensor(tDof, 0, 2) = tZ0(tDof);
      tTensor(tDof, 1, 2) = tZ1(tDof);
      tTensor(tDof, 2, 2) = tZ2(tDof);
    }, "create tensor");

    auto tHost = Kokkos::create_mirror_view(tTensor);
    Kokkos::deep_copy(tHost, tTensor);
    return tTensor;
  }

  Plato::ScalarArray3D
  createOrthonormalTensor(
    Teuchos::ParameterList & tTransformParams,
    Plato::DataMap         & aDataMap,
    Plato::OrdinalType       aDimension
  )
  {
    if(aDimension == 3)
    {
      return createOrthonormalTensor3D(tTransformParams, aDataMap);
    }
    else
    if(aDimension == 2)
    {
      return createOrthonormalTensor2D(tTransformParams, aDataMap);
    }
    else
    if(aDimension == 1)
    {
      ANALYZE_THROWERR("createOrthonormalTensor not implemented for 1D.");
    }

    ANALYZE_THROWERR("Invalid dimension.");
  }

  void
  readInputData(
    Teuchos::ParameterList & aInputs,
    Plato::DataMap         & aDataMap,
    Plato::Mesh              aMesh
  )
  {
    
    if(aInputs.isSublist("Input Data") == false) return;

    auto tInputDataParams = aInputs.sublist("Input Data");

    for (auto tItr = tInputDataParams.begin(); tItr != tInputDataParams.end(); ++tItr)
    {
      const auto & tEntry = tInputDataParams.entry(tItr);
      if (!tEntry.isList())
      {
        ANALYZE_THROWERR("Parameter in 'Input Data' block not allowed.  Expect lists only.")
      }

      const std::string &tName = tInputDataParams.name(tItr);
      auto & tSubList = tInputDataParams.sublist(tName);

      // creation/reading of input data could eventually be done by a factory
      //
      if(tSubList.isType<std::string>("Input File") == false)
      {
        ANALYZE_THROWERR("'Input File' not specified.")
      }

      auto tInputFileName = tSubList.get<std::string>("Input File");
    
      auto tExtension = Plato::getFileExtension(tInputFileName);

      if( Plato::tolower(tExtension) == "csv" )
      {
        readCSVData(tInputFileName, tSubList, aDataMap, aMesh);
      }
      else
      {
        std::stringstream ss;
        ss << "Unrecognized file extension: " << tExtension;
        ANALYZE_THROWERR(ss.str());
      }

      // apply transform if requested
      if(tSubList.isSublist("Transform"))
      {
        auto tTransformParams = tSubList.sublist("Transform");
        auto tType = Plato::ParseTools::getParam<std::string>(tTransformParams, "Type");
        auto tFieldName = Plato::ParseTools::getParam<std::string>(tTransformParams, "Field Name");

        if(tType == "Orthonormal Tensor")
        {
          if(aDataMap.scalarArray3Ds.count(tFieldName))
          {
            std::stringstream ss;
            ss << "Attempted to add a field '" << tFieldName << "' that already exists.";
            ANALYZE_THROWERR(ss.str())
          }
          auto tDimension = aMesh->NumDimensions();
          aDataMap.scalarArray3Ds[tFieldName] = createOrthonormalTensor(tTransformParams, aDataMap, tDimension);
        }
      }
    }
  }
}
