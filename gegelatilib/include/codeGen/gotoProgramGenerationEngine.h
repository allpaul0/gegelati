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
#ifndef GOTO_PROGRAM_GENERATION_ENGINE_H
#define GOTO_PROGRAM_GENERATION_ENGINE_H

#include "codeGen/programGenerationEngine.h"

namespace CodeGen {

/**
 * \brief Specialisation of ProgramGenerationEngine for the computed-goto
 * dispatch style. 
 *
 * Runtime polymorphism (dynamic dispatch) is used so that a
 * ProgramGenerationEngine base-class pointer or reference can refer to a
 * GotoProgGenEngine object and invoke the appropriate overridden virtual
 * methods at runtime.
 *
 * Key design decisions driven by the base class implementation:
 *
 *  1. We want to generate header-only code (all program bodies in fileH) so that
 *     GCC can inline it.  The base class is designed to write program bodies to
 *     fileC, so we must redirect that output to fileH.
 * 
 *     gotoProgramGenerationEngine::openFile() overrides the base class method.
 *     It opens fileH normally and fileC
 *     on a platform-null device (e.g. /dev/null) so it has a valid stream.
 *     
 *    2. generateProgram() is overriden to emit
 *     "inline __attribute__((always_inline)) fixedpt P<id>(...)" into
 *     fileH instead of the base "double P<id>()" into fileC.
 *     
 *     We redirect the fileC stream to fileH and call base class methods
 *     for program generation on fileC before restoring the fileC stream.
 *  
 *     iterateThroughtProgram() -> processLine() 
 *     -> generateCurrentLine() -> initOperandCurrentLine() 
 *     are all base class methods that write to fileC. 
 * 
 *      This design allows us to reuse the base class's program generation logic.
 *
 *  3. getNameSourceData() is overriden so that data-source indices map
 *     to function parameters "inN" instead of global extern pointers. Callers must hold a
 *     GotoProgramGenerationEngine* for dispatch to work correctly.
 *
 */

  class GotoProgramGenerationEngine : public ProgramGenerationEngine
  {
    public:
      /**
       * \param filename   Base name; the engine writes to <filename>.h.
       * \param env        The GEGELATI Environment (registers, constants, ISet, DataHandlers, …).
       * \param path       Output directory (trailing '/' required).
       * \param globalVarUsed Whether to use global variables to access data sources
       * \param nbInputs   Number of "const fixedpt * restrict inN" parameters
       *                   (default 4).
       */
      GotoProgramGenerationEngine(const std::string& filename,
                                  const Environment& env,
                                  const std::string& path = "./",
                                  CodeGen::Dtype Dtype = CodeGen::Dtype::Double,
                                  bool globalVarUsed = false,
                                  int nbInputs = 4);

      /**
       * \brief Destructor.
       * does nothing.
       */
      ~GotoProgramGenerationEngine() override;

      /**
       * \brief Generates one inline program function into fileH.
       *
       * overrides ProgramGenerationEngine::generateProgram() — callers must
       * use a GotoProgramGenerationEngine* or reference.
       *
       * While the program lines are being generated (iterateThroughtProgram)
       * the stream buffer of fileC is temporarily redirected to fileH so
       * that generateCurrentLine() / initOperandCurrentLine() output lands
       * in the header.
       *
       * \param progID          Unique identifier for the program.
       * \param ignoreException Forwarded to iterateThroughtProgram().
       */
      void generateProgram(uint64_t progID, bool ignoreException) override;

    protected:
      /**
       * brief Maps data-source indices to function parameter names.
       *
       * Shadows ProgramGenerationEngine::getNameSourceData().
       *  idx == 0                        → "reg"
       *  idx == 1, nbProgramConstant > 0 → "cst"
       *  otherwise                       → "in1", "in2", …
       */
    //  std::string getNameSourceData(const uint64_t& idx) override;

    private:
      /// Number of "const fixedpt * restrict inN" parameters.
      int nbInputs;

      /**
       * \brief Overrides ProgramGenerationEngine::openFile() to implement 
       *        the goto-sstyle.
       *        Initialises fileH with the goto-style header prologue and
       *        opens fileC on /dev/null so iterateThroughtProgram() has a
       *        valid (but discarded) sink before the redirect is installed.
       *
       * \param filename  filename 
       * \param path      Output directory.
       * \param nbConstant Number of program constants (used for include guard).
       */
      void openFile(const std::string& filename, 
                    const std::string& path,
                    size_t nbConstant) override;
  };

} // namespace CodeGen

#endif // GOTO_PROGRAM_GENERATION_ENGINE_H
#endif // CODE_GENERATION