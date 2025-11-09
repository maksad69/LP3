// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract StudentData {

    struct S {
        uint id;
        string f;
        string l;
        uint8 a;
        string c;
    }

    S[] private ss;
    mapping(uint => uint) private idx;

    uint public cnt;
    uint public td;
    uint public lda;
    address public ldb;

    event SA(uint indexed id, string f);
    event SU(uint indexed id);
    event SR(uint indexed id);
    event ER(address indexed from, uint amount);
    event FC(address indexed from, uint amount, bytes data);

    function add(string calldata _f, string calldata _l, uint8 _a, string calldata _c) external {
        uint newId = ++cnt;
        ss.push(S(newId, _f, _l, _a, _c));
        idx[newId] = ss.length;
        emit SA(newId, _f);
    }

    function get(uint _id) external view returns (S memory) {
        uint p = idx[_id];
        require(p != 0, "Not found");
        return ss[p - 1];
    }

    function getAll() external view returns (S[] memory) {
        return ss;
    }

    function update(uint _id, string calldata _f, string calldata _l, uint8 _a, string calldata _c) external {
        uint p = idx[_id];
        require(p != 0, "Not found");
        S storage s = ss[p - 1];
        s.f = _f;
        s.l = _l;
        s.a = _a;
        s.c = _c;
        emit SU(_id);
    }

    function remove(uint _id) external {
        uint p = idx[_id];
        require(p != 0, "Not found");

        uint i = p - 1;
        uint last = ss.length - 1;

        if (i != last) {
            S memory ls = ss[last];
            ss[i] = ls;
            idx[ls.id] = i + 1;
        }

        ss.pop();
        delete idx[_id];

        emit SR(_id);
    }

    receive() external payable {
        td++;
        lda = msg.value;
        ldb = msg.sender;
        emit ER(msg.sender, msg.value);
    }

    fallback() external payable {
        td++;
        lda = msg.value;
        ldb = msg.sender;
        emit FC(msg.sender, msg.value, msg.data);
    }

    function getCount() external view returns (uint) {
        return ss.length;
    }
}
