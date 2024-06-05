/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2022) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2022)
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

#include "tpg/instrumented/tpgFactoryInstrumented.h"
#include "tpg/instrumented/tpgActionInstrumented.h"
#include "tpg/instrumented/tpgEdgeInstrumented.h"
#include "tpg/instrumented/tpgExecutionEngineInstrumented.h"
#include "tpg/instrumented/tpgTeamInstrumented.h"

std::shared_ptr<TPG::TPGGraph> TPG::TPGFactoryInstrumented::createTPGGraph(const Environment& env) const
{
    return std::make_shared<TPG::TPGGraph>(env, std::make_unique<TPGFactoryInstrumented>());
}

TPG::TPGTeam* TPG::TPGFactoryInstrumented::createTPGTeam() const
{
    return new TPGTeamInstrumented();
}

TPG::TPGAction* TPG::TPGFactoryInstrumented::createTPGAction(const uint64_t id) const
{
    return new TPGActionInstrumented(id);
}

std::unique_ptr<TPG::TPGEdge> TPG::TPGFactoryInstrumented::createTPGEdge(
    const TPGVertex* src, const TPGVertex* dest,
    const std::shared_ptr<Program::Program> prog) const
{
    auto ptr = std::make_unique<TPG::TPGEdgeInstrumented>(src, dest, prog);
    return ptr;
}

std::unique_ptr<TPG::TPGExecutionEngine> TPG::TPGFactoryInstrumented::
    createTPGExecutionEngine(const Environment& env, Archive* arch) const
{
    return std::make_unique<TPGExecutionEngineInstrumented>(env, arch);
}

void TPG::TPGFactoryInstrumented::resetTPGGraphCounters(
    const TPG::TPGGraph& tpgGraph) const
{
    // Reset all vertices
    for (const TPG::TPGVertex* vertex : tpgGraph.getVertices()) {
        const TPG::TPGVertexInstrumented* vertexI = dynamic_cast<const TPG::TPGVertexInstrumented*>(vertex);
        if (vertexI != nullptr) {
            vertexI->reset();
        }
    }

    // Reset all edges
    for (const auto& edge : tpgGraph.getEdges()) {
        const TPG::TPGEdgeInstrumented* edgeI = dynamic_cast<const TPG::TPGEdgeInstrumented*>(edge.get());
        if (edgeI != nullptr) {
            edgeI->reset();
        }
    }
}

void TPG::TPGFactoryInstrumented::clearUnusedTPGGraphElements(TPG::TPGGraph& tpgGraph) const
{
    // Remove unused vertices first
    // (this will remove a few edges as a side-effect)
    // Work on a copy of vertex list as the graph is modified during the for
    // loop.
    std::vector<const TPG::TPGVertex*> vertices(tpgGraph.getVertices());
    for (const TPG::TPGVertex* vertex : vertices) {
        
        const TPG::TPGVertexInstrumented* vertexI = dynamic_cast<const TPG::TPGVertexInstrumented*>(vertex);
        
        // If the vertex is instrumented AND was never visited
        if (vertexI != nullptr && vertexI->getNbVisits() == 0) {
            // remove it
            tpgGraph.removeVertex(*vertex);
        }
    }

    // Remove un-traversed edges
    std::vector<const TPG::TPGEdge*> edges;

    // Copy the edge list before iteration
    for (const std::unique_ptr<TPG::TPGEdge>& edgePtr : tpgGraph.getEdges()) {
        
        // Dereference the unique_ptr to access TPGEdge object
        const TPG::TPGEdge& edge = *edgePtr;
        edges.push_back(&edge);
    }
    
    // Iterate on the edge list
    for (const TPG::TPGEdge* edge : edges) {
       
        const TPG::TPGEdgeInstrumented* edgeI = dynamic_cast<const TPG::TPGEdgeInstrumented*>(edge);
        
        // If the vertex is instrumented AND was never visited
        if (edgeI != nullptr && edgeI->getNbTraversal() == 0) {
            // remove it
            tpgGraph.removeEdge(*edge);
        }
    }

    // Remove teams with only one output program
    std::vector<const TPG::TPGVertex*> vertices2(tpgGraph.getVertices());
    
    for (const TPG::TPGVertex* vertex : vertices2) {
        
        // If vertex is a team
        if (typeid(*vertex) == typeid(TPG::TPGTeamInstrumented)){
            
            const TPG::TPGTeamInstrumented* teamI = dynamic_cast<const TPG::TPGTeamInstrumented*>(vertex);
            
            // If the team has only one outgoing edge
            if(teamI->getOutgoingEdges().size() == 1){
                
                // Get the destination vertex of the team
                auto destinationVertices = teamI->getOutgoingEdges().front()->getDestination();
                
                // Set the new destination of all the incomming edges to the new destination
                std::vector<TPG::TPGEdge *> incomingEdges;
                
                for(TPG::TPGEdge* incomingEdge: teamI->getIncomingEdges()){
                    incomingEdges.push_back(incomingEdge);
                }
                
                for(TPG::TPGEdge* incomingEdge: incomingEdges){
                    tpgGraph.setEdgeDestination(*incomingEdge, *destinationVertices);
                }
                
                // Delete the program between old and new destination ?
                tpgGraph.removeVertex(*vertex);
            }
        }
    }
}
