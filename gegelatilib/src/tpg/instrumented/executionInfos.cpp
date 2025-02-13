/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2022) :
 *
 * Elinor Montmasson <elinor.montmasson@gmail.com> (2022)
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

#include "tpg/instrumented/executionInfos.h"

#include <algorithm>
#include <fstream>
#include <json.h>
#include <numeric>
#include <vector>
#include <queue>
#include <string>

#include "program/program.h"
#include "tpg/instrumented/tpgTeamInstrumented.h"
#include "tpg/instrumented/tpgActionInstrumented.h"
#include "tpg/instrumented/tpgEdgeInstrumented.h"
#include "tpg/instrumented/tpgVertexInstrumented.h"
#include "tpg/instrumented/tpgFactoryInstrumented.h"

void TPG::ExecutionInfos::analyzeProgram(
    std::map<uint64_t, uint64_t>& nbExecutionForEachInstr,
    const Program::Program& program)
{

    for (int i = 0; i < program.getNbLines(); i++) {
        nbExecutionForEachInstr[program.getLine(i).getInstructionIndex()]++;
    }
}

void TPG::ExecutionInfos::analyzeInferenceTrace(const std::vector<const TPGVertex*>& inferenceTrace, unsigned int seed)
{
    // Do not count the action vertex at the end
    uint64_t nbEvaluatedTeams = inferenceTrace.size() - 1;
    uint64_t nbEvaluatedPrograms = 0;
    uint64_t nbExecutedLines = 0;
    std::map<uint64_t, uint64_t> nbExecutionForEachInstr;
    std::list<int> traceTeamIds;

    // For of each visited teams, analysing its edges
    for (std::vector<const TPG::TPGVertex *>::const_iterator inferenceTraceTeamIterator = inferenceTrace.cbegin(); 
        inferenceTraceTeamIterator != inferenceTrace.cend() - 1; inferenceTraceTeamIterator++) {

        // retrieve Team id and add it to the list of traces
        traceTeamIds.push_back(((TPG::TPGTeamInstrumented*) *inferenceTraceTeamIterator)->getId());

        for (const TPG::TPGEdge* edge : (*inferenceTraceTeamIterator)->getOutgoingEdges()) {
  
            // Edges leading to a previously visited teams (including the current team) are not evaluated
            auto endSearchIt = inferenceTraceTeamIterator + 1;

            if (std::find(inferenceTrace.cbegin(), endSearchIt, edge->getDestination()) != endSearchIt) {
                throw std::logic_error("Error: Edge leads to a previously visited vertex in a DAG.");
                //continue; 
            }

            nbEvaluatedPrograms++;
            nbExecutedLines += edge->getProgram().getNbLines();
            analyzeProgram(nbExecutionForEachInstr, edge->getProgram());
        }
    }

    this->vecInferenceTraceInfos.push_back({seed, nbEvaluatedTeams, nbEvaluatedPrograms, nbExecutionForEachInstr, traceTeamIds});
}

void TPG::ExecutionInfos::analyzeExecution(
    TPG::TPGExecutionEngineInstrumented& tee, const TPG::TPGGraph& tpgGraph, unsigned int seed)
{
    // 1. Get nbExecutionForEachInstr, nbEvaluatedTeams, nbEvaluatedPrograms

    // Retrieve inference trace history (every inference of a TPG leads to a trace: vector of visited vertices)
    const auto& traceHistory = tee.getInferenceTraceHistory();

    // Raise an error if the size of the vector is greater than 1, we are not using this method as intended
    if (traceHistory.size() > 1) {
        throw std::runtime_error("Error: The size of inferenceTraceHistory is greater than 1.");
    }

    const std::vector<const TPG::TPGVertex*>& inferenceTrace = traceHistory.front();
    analyzeInferenceTrace(inferenceTrace, seed);
        
    // 2. clear Trace History, clear TPGGraph
    // Clear the trace history from all previous inference trace.
    // Execution Stats uses this trace to generate statistics of inference.
    tee.clearInferenceTraceHistory();

    // Reset all visit and traversal counters of a TPGGraph, we are measuring for one action on one seed
    const TPG::TPGFactoryInstrumented* factoryInstrumented = 
        dynamic_cast<const TPG::TPGFactoryInstrumented*>(&tpgGraph.getFactory());
    
    if (!factoryInstrumented) {
        throw std::runtime_error("Error: TPGFactory is not of type TPGFactoryInstrumented.");
    }

    factoryInstrumented->resetTPGGraphCounters(tpgGraph);
}

void TPG::ExecutionInfos::writeTPGtoJson(const char* filePath, bool noIndent) const
{
    std::ofstream outputFile(filePath);

    Json::Value root;

    for (auto it : this->nbProgsbyTeam) {
        root["tpg"][std::to_string(it.first)] = it.second;
    }

    Json::StreamWriterBuilder writerFactory;
    // Set a precision to 6 digits after the point.
    writerFactory.settings_["precision"] = 6U;
    if (noIndent)
        writerFactory.settings_["indentation"] = "";
    Json::StreamWriter* writer = writerFactory.newStreamWriter();
    writer->write(root, &outputFile);
    delete writer;

    outputFile.close();
}

void TPG::ExecutionInfos::writeInfosToJson(const char* filePath, bool noIndent) const
{
    
    std::ofstream outputFile(filePath);

    Json::Value root;

    // Trace infos
    int i = 0;
    for (TPG::InferenceTraceInfos inferenceTraceInfos: this->vecInferenceTraceInfos) {
        //std::string i_str = std::to_string(i);
        std::string seed_str = std::to_string(inferenceTraceInfos.seed);

        //root[i_str]["seed"] = inferenceTraceInfos.seed;
        root[seed_str]["nbEvaluatedTeams"] = inferenceTraceInfos.nbEvaluatedTeams;
        root[seed_str]["nbEvaluatedPrograms"] = inferenceTraceInfos.nbEvaluatedPrograms;

        for (const auto& pair : inferenceTraceInfos.nbExecutionForEachInstr) {
            uint64_t key = pair.first;     // The key in the map
            uint64_t value = pair.second;  // The corresponding value in the map

            std::string key_str = std::to_string(key);

            root[seed_str]["nbExecutionForEachInstr"][key_str] = value;
        }
        i++;

        std::list<int>::iterator it;
        std::string fullTrace = "";
        for (it = inferenceTraceInfos.traceTeamIds.begin(); it != inferenceTraceInfos.traceTeamIds.end(); ++it){
            fullTrace = fullTrace + std::to_string(*it) + ", ";
        }
        fullTrace = fullTrace.substr(0, fullTrace.size()-2);
        root[seed_str]["traceTeamIds"] = fullTrace;
    }

    Json::StreamWriterBuilder writerFactory;
    // Set a precision to 6 digits after the point.
    writerFactory.settings_["precision"] = 6U;
    if (noIndent)
        writerFactory.settings_["indentation"] = "";
    Json::StreamWriter* writer = writerFactory.newStreamWriter();
    writer->write(root, &outputFile);
    delete writer;

    outputFile.close();
    
}

void TPG::ExecutionInfos::assignIdentifiers(const TPG::TPGTeamInstrumented* root) {

    // safety check
    if (root == NULL) 
        return;

    // Create an empty queue for level order traversal
    std::queue<const TPG::TPGTeamInstrumented*> q;
    // Create an empty set to make sure each Vertex is counted only once
    std::set<const TPG::TPGTeamInstrumented*> s;
    int currentId = 0;

    q.push(root);
    s.insert(root);

    while (q.empty() == false) {

        // Assign (could be printed) front of queue and remove it
        const TPG::TPGTeamInstrumented* vertex = q.front();
        vertex->setId(currentId++);
        std::cout << "Id: " << vertex->getId() << ", nbOutGoingEdges: " << vertex->getOutgoingEdges().size() << std::endl;
        this->nbProgsbyTeam[vertex->getId()] = vertex->getOutgoingEdges().size();
        q.pop();

        // Enqueue every child of this team (teams pointed by OutgoingEdges)
        for (const TPG::TPGEdge* edge: vertex->getOutgoingEdges()){             
            const TPG::TPGTeamInstrumented* destinationTeam = dynamic_cast<const TPG::TPGTeamInstrumented*>(edge->getDestination());
            if(destinationTeam && !s.count(destinationTeam)){ // make sure we don't add duplicate in our queue
                q.push(destinationTeam);
                s.insert(destinationTeam);
            }
        }
    }
}