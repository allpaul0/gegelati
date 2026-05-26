/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2026) :
 * 
 * Paul Allaire <paul.allaire@insa-rennes.fr> (2026)
 *
 * GEGELATI is an open-source reinforcement learning framework for training
 * artificial intelligence based on Tangled Program Graphs (TPGs).
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 */

#ifdef CODE_GENERATION
#ifndef DTYPE_H
#define DTYPE_H

/** 
* \brief
* Utility class to represent the data type of the generated code. 
* The TPGGenerationEngineFactory create() method sets the data type of the 
* generated code (default double) by instanciating a TPGGenerationEngine 
* that stores the data type as a member variable.
* 
* This class provides serilization methods to convert between the data type
* defined in the enum and their string representations.
**/

namespace CodeGen {
  enum class Dtype
  {
    Double,
    Float,
    Fixedpt,
    Int
  };

  constexpr std::string_view to_string(Dtype dtype_val){
    switch (dtype_val)
    {
      case Dtype::Double : return "double"; 
      case Dtype::Float : return "float"; 
      case Dtype::Fixedpt : return "fixedpt"; 
      case Dtype::Int : return "int";
    } 
    return "unknown_type";
  }

  constexpr CodeGen::Dtype dtype_from_string(std::string_view dtype_string) {
    if (dtype_string == "double") {
        return Dtype::Double;
    } else if (dtype_string == "float") {
        return Dtype::Float;
    } else if (dtype_string == "fixedpt") {
        return Dtype::Fixedpt;
    } else if (dtype_string == "int") {
        return Dtype::Int;
    }

    throw std::invalid_argument("Unknown Dtype");
  }

  inline std::ostream& operator<<(std::ostream& os, Dtype type) {
    return os << to_string(type);
  }

} // namespace CodeGen

#endif // DTYPE_H
#endif // CODE_GENERATION