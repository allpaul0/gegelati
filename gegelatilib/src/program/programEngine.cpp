/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2023) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2023)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2019 - 2020)
 * Thomas Bourgoin <tbourgoi@insa-rennes.fr> (2021)
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

#include "program/programEngine.h"
#include "data/constantHandler.h"

void Program::ProgramEngine::setProgram(const Program& prog)
{
    // are constants used here ?
    size_t offset = 1;
    if (prog.getEnvironment().getNbConstant() > 0) {
        // replace programs constants if already existing
        dataScsConstsAndRegs.at(1) = prog.cGetConstantHandler();
        // increment offset for the datahandlers verification
        offset++;
    }

    // Check dataSource are similar in all point to the program environment
    // offset is -1 if there is only the registers to ignore
    // -2 because we don't count the registers that are the first datasources
    // and the constants (second datasource)
    if (this->dataScsConstsAndRegs.size() - offset != prog.getEnvironment().getDataSources().size()) {
        throw std::runtime_error(
            "Data sources characteristics for Program Execution differ from "
            "Program reference Environment.");
    }
    for (size_t i = 0; i < this->dataScsConstsAndRegs.size() - offset; i++) {
        // check data source characteristics
        auto& iDataSrc = this->dataScsConstsAndRegs.at(i + (size_t)offset).get();
        auto& envDataSrc = prog.getEnvironment().getDataSources().at(i).get();
        // Assume that dataSource must be (at least) a copy of each other to
        // simplify the comparison This is characterise by the two data sources
        // having the same id
        if (iDataSrc.getId() != envDataSrc.getId()) {
            throw std::runtime_error(
                "Data sources characteristics for Program Execution differ "
                "from Program reference Environment.");
            // If this pose a problem one day, an additional more
            // complex check could be used as a last resort when ids
            // of DataHandlers are different: checking equality of the
            // lists of provided data types and the equality address
            // space size for each data type.
        }
    }
    // set the program
    this->program = &prog;
    // Reset Registers (in case it is not done when they are constructed)
    this->registers.resetData();

    // Reset the counters
    this->programCounter = 0;
}

const std::vector<std::reference_wrapper<const Data::DataHandler>>& Program::ProgramEngine::getDataSources() const
{
    return this->dataSources;
}

const bool Program::ProgramEngine::next()
{
    // While the next line is an intron
    // increment the program counter.
    do {
        this->programCounter++;
    } while (this->programCounter < this->program->getNbLines() && this->program->isIntron(this->programCounter));
    return this->programCounter < this->program->getNbLines();
}

const Program::Line& Program::ProgramEngine::getCurrentLine() const
{
    return this->program->getLine(this->programCounter);
}

const Instructions::Instruction& Program::ProgramEngine::getCurrentInstruction()
    const
{
    const Line& currentLine =
        this->getCurrentLine(); // throw std::out_of_range if the program
    // counter is too large.
    uint64_t instructionIndex = currentLine.getInstructionIndex();
    return this->program->getEnvironment().getInstructionSet().getInstruction(
        instructionIndex); // throw std::out_of_range if the index of the line
    // is too large.
}

const void Program::ProgramEngine::fetchCurrentOperands(
    std::vector<Data::UntypedSharedPtr>& operands) const
{
    const Line& line = this->getCurrentLine(); // throw std::out_of_range
    const Instructions::Instruction& instruction =
        this->getCurrentInstruction(); // throw std::out_of_range

    // Get as many operands as required by the instruction.
    for (uint64_t i = 0; i < instruction.getNbOperands(); i++) {
        const Data::DataHandler& dataSource = this->dataScsConstsAndRegs.at(
            line.getOperand(i).first); // Throws std::out_of_range
        const uint64_t operandLocation = getOperandLocation(i);
        const std::type_info& operandType =
            instruction.getOperandTypes().at(i).get();
        Data::UntypedSharedPtr data =
            dataSource.getDataAt(operandType, operandLocation);
        operands.push_back(data);
    }
}

uint64_t Program::ProgramEngine::getOperandLocation(uint64_t idxOp) const
{
    const Line& line = this->getCurrentLine(); // throw std::out_of_range
    const Instructions::Instruction& instruction =
        this->getCurrentInstruction(); // throw std::out_of_range

    const std::pair<uint64_t, uint64_t>& operandIndexes =
        line.getOperand(idxOp);
    const Data::DataHandler& dataSource = this->dataScsConstsAndRegs.at(
        operandIndexes.first); // Throws std::out_of_range
    const std::type_info& operandType =
        instruction.getOperandTypes().at(idxOp).get();
    uint64_t operandLocation =
        dataSource.scaleLocation(operandIndexes.second, operandType);

    return operandLocation;
}
void Program::ProgramEngine::iterateThroughtProgram(const bool ignoreException)
{
    this->programCounter = 0;
    bool hasNext = this->program->getNbLines() > 0;

    // Skip first lines if they are introns.
    if (hasNext && this->program->isIntron(0)) {
        hasNext = this->next();
    }

    // Execute useful lines
    while (hasNext) {
        try {
            // process the current line
            processLine();
        }
        catch (std::out_of_range& e) {
            if (!ignoreException) {
                throw e; // rethrow
            }
        }

        // Increment the programCounter.
        hasNext = this->next();
    };
}
