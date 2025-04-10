import pytest

from ..src.dlx import Sudoku
from ..src.backtracking import Sudoku as BacktrackingSudoku


@pytest.fixture()
def sudoku9x9():
    return "800000000003600000070090200050007000000045700000100030001000068008500010090000400"

@pytest.fixture()
def sudoku9x9solution():
    return "812753649943682175675491283154237896369845721287169534521974368438526917796318452"

@pytest.fixture()
def sudoku16x16():
    return "C21F000A0009064000009000AD050020600304E01000D0A870AB5003F624G9C00C00A00E6030F0B000051G089000C00000G06030501F2000380002BDE040650G0060B01024C80GD390D24060700B000CB1C008000093E0708007DE90G006400200E680403750AC0B00000500CF020000500CEFG1B0000D67130A0B004GD0028F"

@pytest.fixture()
def sudoku16x16solution():
    return "C21FG7DA83E9B6454E8G96CFADB57321695324EB1CG7DFA87DAB5183F624G9CE2C41A95E683GF7BDE6B51GF8927DC43AA7GD6C345B1F28E938F972BDEA4C651GF56EBA1724C89GD39GD243657EAB81FCB1C4F82GD593EA768A37DE9CG1F64B52GFE68D423751AC9BDB7835A9CF621EG4542CEFG1B98A3D67139ACB764GDE528F"

@pytest.fixture()
def sudoku16x16_2():
    return "00000900A000020000A60000450900ED0034280E06700F00000C000D031F0467G000C392085B0010006A00080E000DF000000000C090070E0B9210A00000005840E00B09300070D0A3080C1000G600000000000050F08004000004000000060100D03070000E009000G0600B000D002006000080B10000700E0B0GC070050100"

@pytest.fixture()
def sudoku16x16_2solution():
    return "751DF946ABEG328C2FA6713C4589GBEDB93428GED67C1FA5EG8CA5BD231F9467GD7EC392F85B4A163C6AG7581E24BDF95841B6DFCA9327GEFB921EA46GD7C35841EF8B693CA275DGA358DC1794G6FEB267BGEA235DF189C4D2C954FGE7B8A631CAD53271GF4E689B14G76FEB893D5C2A96234D85B1CAEG7F8EFB9GCA7265D143"


def test_backtracking_9x9(sudoku9x9, sudoku9x9solution):
    sudoku = BacktrackingSudoku(sudoku9x9)
    sudoku.solve()
    assert sudoku.is_solution()
    assert sudoku9x9solution == sudoku.get_state()
    
def test_backtracking_16x16(sudoku16x16, sudoku16x16solution):
    sudoku = BacktrackingSudoku(sudoku16x16)
    sudoku.solve()
    assert sudoku.is_solution()
    assert sudoku16x16solution == sudoku.get_state()
    
def test_dlx_9x9(sudoku9x9, sudoku9x9solution):
    sudoku = Sudoku(sudoku9x9)
    result = sudoku.solve()
    for solution in result:
        break
    is_valid = BacktrackingSudoku(sudoku9x9)
    is_valid.set_solution(solution)
    assert is_valid.is_solution()
    assert sudoku9x9solution == "".join(solution)
    
def test_dlx_16x16(sudoku16x16, sudoku16x16solution):
    sudoku = Sudoku(sudoku16x16)
    result = sudoku.solve()
    for solution in result:
        break
    is_valid = BacktrackingSudoku(sudoku16x16)
    is_valid.set_solution(solution)
    assert is_valid.is_solution()
    assert sudoku16x16solution == "".join(solution)
    
def test_dlx_16x16_2(sudoku16x16_2, sudoku16x16_2solution):
    sudoku = Sudoku(sudoku16x16_2)
    result = sudoku.solve()
    for solution in result:
        break
    is_valid = BacktrackingSudoku(sudoku16x16_2)
    is_valid.set_solution(solution)
    assert is_valid.is_solution()
    assert sudoku16x16_2solution == "".join(solution)

