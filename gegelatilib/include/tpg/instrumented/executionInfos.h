/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2024) :
 *
 * Paul Allaire <paul.allaire@insa-rennes.fr> (2024)
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

#ifndef EXECUTION_INFOS_H
#define EXECUTION_INFOS_H

#include "tpg/instrumented/tpgExecutionEngineInstrumented.h"
#include "tpg/tpgGraph.h"

namespace TPG {

    // inference trace is a keyword for visited vertices from one inference.
    // As the inference of a TPG varies according to its execution context,
    // inference traces are the history of the execution path of a TPG from
    // the starting root/Team, passing by every visited vertices/Team to the terminal 
    // leaf/action. 

    /**
     * \brief Store execution informations of one inference trace.
     *
     * It contains :
     * - the inference trace in a std::vector<const TPG::TPGVertex*>
     * - the number of evaluated teams
     * - the number of evaluated programs
     * - the number of executed lines
     * - the number of execution for each instructions (indexed by instruction
     * index)
     */

        struct InferenceTraceInfos
        {
            /// The seed corresponding to the inference Trace
            const uint64_t seed;
            /// The inference trace.
            //const std::vector<const TPG::TPGVertex*> inferenceTrace;
            /// Number of team evaluated.      // NbVisits
            const uint64_t nbEvaluatedTeams; 
            /// Number of programs evaluated.  //NbTraversals
            const uint64_t nbEvaluatedPrograms;
            /// Map that associate the instruction indexes with the number of
            /// execution of the corresponding Instruction.
            const std::map<uint64_t, uint64_t> nbExecutionForEachInstr;
            /// identifiers for each traversed Team
            std::list<int> traceTeamIds;
        };

     /**
     * \brief Utility class for extracting execution informations
     * from a TPGExecutionEngineInstrumented and an instrumented TPGGraph.
     *
     * ********************** TO DO WRITE THIS **********************
     * ********************** TO DO WRITE TESTS **********************

     *
     * The Json exporter is designed to be used after a call to
     * analyzeExecution(). Just call writeInformationsToJson() to export informations in
     * a file with json format.
     */

    class ExecutionInfos
    {

    private:

        /* Analyzed inference traces */

        /// Analyzed inference traces.
        std::vector<InferenceTraceInfos> vecInferenceTraceInfos;

    protected:
        
        /**
         * \brief Analyze a program to get how many times each instruction is
         * used.
         *
         * \param[out] nbExecutionForEachInstr the std::map<uint64_t, uint64_t>& that
         * will be incremented for each instruction use.
         * \param[in] program the analyzed program.
         */
        static void analyzeProgram(
            std::map<uint64_t, uint64_t>& nbExecutionForEachInstr,
            const Program::Program& program);

    public:
        /// Default constructor.
        ExecutionInfos() = default;

        /// Default destructor.
        virtual ~ExecutionInfos() = default;

          /**
         * \brief
         *
         * Results are stored in the average results class attributes.
         *
         * \param[in] graph the analyzed TPGGraph*.
         * \throws std::bad_cast if graph contains at least one non instrumented
         * vertex or edge.
         */
        void analyzeInstrumentedGraph(const TPGGraph* graph);

        /**
         * \brief Analyze the execution informations of one inference trace.
         *
         * The vector inferenceTrace contains all visited vertices for one inference
         * in order : inferenceTrace[0] is the root, and inferenceTrace[inferenceTrace.size()-1] the
         * action.
         * 
         * From a single inference of an instrumented TPGGraph:
         *   - Calculate `nbEvaluatedPrograms`: the total number of evaluated programs.
         *   - Calculate `nbEvaluatedTeams`: the total number of evaluated teams.
         *   - Retrieve `nbExecutionForEachInstr`: a map where each entry represents the total execution count for each instruction.
         *
         * Results are stored in a new inferenceTraceInfos struct which is pushed back
         * in attribute vecInferenceTraceInfos.
         *
         * \param[in] inferenceTrace a vector<const TPGVertex*> of the analyzed inference
         * \param[in] seed the seed that led to an inference trace on the TPGGraph
         */
        void analyzeInferenceTrace(const std::vector<const TPGVertex*>& inferenceTrace, unsigned int seed);

        /**
         * \brief Analyze the execution informations of multiple inferences
         * done with a TPGExecutionEngineInstrumented.
         *
         * Previous results will be erased.
         *
         * \param[in] tee the TPGExecutionEngineInstrumented.
         * \param[in] graph the TPGGraph executed with tee.
         * \param[in] seed the seed that led to an inference trace on the TPGGraph
         * \throws std::bad_cast if the graph contains a non instrumented vertex
         * or edge.
         */
        void analyzeExecution(TPG::TPGExecutionEngineInstrumented& tee,
                            const TPGGraph& graph, unsigned int seed);

          /**
         * \brief ************ TO DO **************
         *
         * \param[in] filePath the path to the output file.
         * \param[in] noIndent true if the json format must not be indented.
         * Files can become large quickly with a lot of inferenceTraces, so if the file
         * will just be analyzed by another program, set this to true to save
         * some space on your disk. Set noIndent to false if you want to keep
         * the file readable.
         */
        void writeInfosToJson(const char* filePath, bool noIndent = false) const;
    

        /**
         * \brief we perfrom a Breadth First Search (BFS) to give a unique identifier 
         * to every Vertices of a trained TPG. 
         * This id can latter be used to see the execution trace of the TPG.
         * id of TPGTeamInstrumented is mutable therefore, this func is const.
         * 
         * The BFS algo uses a queue which makes its complexity = O(N)
         * Reminder: BFS = LEVEL Order traversal 
         *
         * \param[in] root Root vertex of the TPG to annotate.
         */
        void assignIdentifiers(const TPG::TPGTeamInstrumented* root) const;
    };

} // namespace TPG

#endif // EXECUTION_INFOS_H