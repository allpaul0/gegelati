/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2025) :
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
 *        dispatch style.
 *
 * Key design decisions driven by the base class implementation:
 *
 *  1. ProgramGenerationEngine::openFile() no longer opens fileC (it is
 *     commented out in the current source), so fileC is always in a
 *     closed/invalid state.  generateCurrentLine() and
 *     initOperandCurrentLine() both write to fileC, so we must open a
 *     real sink for fileC before calling iterateThroughtProgram(); we use
 *     a platform-null device (/dev/null on POSIX) for this purpose.
 *
 *  2. All program bodies must land in fileH (header-only, so GCC can
 *     inline them).  We achieve this by redirecting fileC's stream buffer
 *     to fileH's buffer while iterateThroughtProgram() runs, then
 *     restoring it afterwards.  The redirect uses std::ostream::rdbuf()
 *     on the std::ostream base of fileC (std::ofstream::rdbuf() is
 *     const; the base class two-argument form is what we need).
 *
 *  3. getNameSourceData() is shadowed (not overridden — base is not
 *     virtual) so that data-source indices map to function parameters
 *     "inN" instead of global extern pointers.  Callers must hold a
 *     GotoProgramGenerationEngine* for dispatch to work correctly.
 *
 *  4. generateProgram() is likewise shadowed to emit
 *     "inline __attribute__((always_inline)) fixedpt P<id>(...)" into
 *     fileH instead of the base "double P<id>()" into fileC.
 *
 *  5. The destructor sets headerClosed = true before the base destructor
 *     runs, preventing a double "#endif" in the output file.
 */
class GotoProgramGenerationEngine : public ProgramGenerationEngine
{
  public:
    /**
     * \param filename   Base name; the engine writes to <filename>.h.
     * \param env        The GEGELATI Environment (registers, constants, …).
     * \param path       Output directory (trailing '/' required).
     * \param nbInputs   Number of "const fixedpt * restrict inN" parameters
     *                   (default 4).
     */
    GotoProgramGenerationEngine(const std::string& filename,
                                const Environment& env,
                                const std::string& path = "./",
                                int nbInputs = 4);

    /**
     * \brief Destructor.
     *
     * Writes the closing #endif guard into fileH, marks headerClosed so
     * the base destructor does not emit a second guard, then closes fileH.
     * fileC (opened on /dev/null) is closed by the base destructor.
     */
    ~GotoProgramGenerationEngine() override;

    /**
     * \brief Generates one inline program function into fileH.
     *
     * Shadows ProgramGenerationEngine::generateProgram() — callers must
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
    void generateProgram(uint64_t progID, bool ignoreException);

  protected:
    /**
     * \brief Maps data-source indices to function parameter names.
     *
     * Shadows ProgramGenerationEngine::getNameSourceData().
     *  idx == 0                        → "reg"
     *  idx == 1, nbProgramConstant > 0 → "cst"
     *  otherwise                       → "in1", "in2", …
     */
    std::string getNameSourceData(const uint64_t& idx);

  private:
    /// Number of "const fixedpt * restrict inN" parameters.
    int nbInputs;

    /**
     * \brief Re-initialises fileH with the goto-style header prologue and
     *        opens fileC on /dev/null so iterateThroughtProgram() has a
     *        valid (but discarded) sink before the redirect is installed.
     *
     * Called from the constructor after the base class has run so we can
     * replace the header content with the goto-style version.
     *
     * \param filename  Base filename (without extension).
     * \param path      Output directory.
     * \param nbConstant Number of program constants (used for include guard).
     */
    void openGotoFile(const std::string& filename, const std::string& path,
                      size_t nbConstant);
};

} // namespace CodeGen

#endif // GOTO_PROGRAM_GENERATION_ENGINE_H
#endif // CODE_GENERATION