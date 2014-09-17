/// <reference path="lib/hammerjs.d.ts" />
/// <reference path="lib/svgjs.d.ts" />
/// <reference path="lib/utils.ts" />
/// <reference path="./model.ts" />
/// <reference path="./view.ts" />
var MIN_VAL = 2;

var grid;
var gridView;
var gridRep = [];
var gameParams;
var gameState;

var canvas = null;

// even-r horizontal layout
var draw = function () {
    canvas.clear();

    // draw cells
    gridRep = [];

    gridView.draw(canvas);
};

var update = function () {
    checkVictoryCondition();
};


// translation of practical flood fill implementation as described on
// http://en.wikipedia.org/wiki/Flood_fill
// this should go into Controller at a much later time
var floodAcquire = function (start, tile) {
    var cluster = [];
    var marked = { get: null, set: null };

    if (grid instanceof Model.Hex) {
        marked.get = function (key) {
            return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
        };
        marked.set = function (key) {
            this[JSON.stringify(key)] = true;
        };
        var Q = [];
        if (grid.get(start.q, start.r) != tile)
            return [];
        Q.push(start);
        while (Q.length > 0) {
            var n = Q.shift();
            if (grid.get(n.q, n.r).value == tile.value && grid.get(n.q, n.r).type == tile.type && !marked.get({ q: n.q, r: n.r })) {
                var w = { q: n.q, r: n.r };
                var e = { q: n.q, r: n.r };

                while (grid.get(w.q - 1, w.r).value == tile.value && grid.get(w.q - 1, w.r).type == tile.type) {
                    w.q--;
                }

                while (grid.get(e.q + 1, e.r).value == tile.value && grid.get(e.q + 1, e.r).type == tile.type) {
                    e.q++;
                }

                for (var q = w.q; q <= e.q; q++) {
                    var nn = { q: q, r: n.r };
                    marked.set({ q: nn.q, r: nn.r });
                    cluster.push(nn);

                    var nw = { q: nn.q, r: nn.r - 1 };
                    var ne = { q: nn.q + 1, r: nn.r - 1 };
                    var sw = { q: nn.q - 1, r: nn.r + 1 };
                    var se = { q: nn.q, r: nn.r + 1 };

                    if (grid.get(nw.q, nw.r).value == tile.value && grid.get(nw.q, nw.r).type == tile.type)
                        Q.push(nw);
                    if (grid.get(ne.q, ne.r).value == tile.value && grid.get(ne.q, ne.r).type == tile.type)
                        Q.push(ne);
                    if (grid.get(sw.q, sw.r).value == tile.value && grid.get(sw.q, sw.r).type == tile.type)
                        Q.push(sw);
                    if (grid.get(se.q, se.r).value == tile.value && grid.get(se.q, se.r).type == tile.type)
                        Q.push(se);
                }
            }
        }
    }
    return cluster;
};

var extendUI = function () {
    var cells = gridView.getDOMElements();

    cells.forEach(function (cell, i) {
        if (cell === null)
            return;
        var hammer = Hammer(cell.node, { preventDefault: true });
        hammer.on("dragstart swipestart", function (e) {
            if (grid.get(cell.data.q, cell.data.r).type != 4 /* DEACTIVATED */) {
                var target = { r: cell.data.r, q: cell.data.q };
                gameState.selected = floodAcquire(target, grid.get(target.q, target.r));
            } else {
                gameState.selected = [];
            }
        });
    });
};

var advance = function () {
    init();
};

var checkVictoryCondition = function () {
    if (gameState.activeCells == 0) {
        advance();
    }
};

var prune = function (start) {
    // see if we should delete this cell and surrounding cells
    var startTile = grid.get(start.q, start.r);
    var group = floodAcquire(start, startTile);
    if (group.length == startTile.value) {
        if (startTile.type == 4 /* DEACTIVATED */)
            return;
        group.forEach(function (cell) {
            // each Tile should have its own "prune" behavior
            if (grid.get(cell.q, cell.r).type == 2 /* REGULAR */) {
                grid.set(cell.q, cell.r, new Tile(1 /* EMPTY */, -1));
                gameState.activeCells--;
            } else if (grid.get(cell.q, cell.r).type == 5 /* LAVA */) {
                grid.set(cell.q, cell.r, new Tile(4 /* DEACTIVATED */, grid.get(cell.q, cell.r).value));
            }
        });
    }
};

var onDrag = function (e) {
    var selected = gameState.selected;
    if (selected == null || selected.length == 0) {
        return;
    }

    var nw = false, ne = false, west = false, east = false, sw = false, se = false;

    var n_vec = { x: 0, y: -1 }, ne_vec = { x: Math.sqrt(3) / 2, y: -0.5 }, se_vec = { x: Math.sqrt(3) / 2, y: 0.5 }, s_vec = { x: 0, y: 1 }, sw_vec = { x: -Math.sqrt(3) / 2, y: 0.5 }, nw_vec = { x: -Math.sqrt(3) / 2, y: -0.5 };

    // returns whether or not B is between A and C
    // http://stackoverflow.com/a/17497339
    // if A cross C and A cross B are in the same direction, then we have either A->C->B or A->B->C
    // if C cross A and C cross B are in the same direction, then we have either C->A->B or C->B->A
    // so we must have A->B->C (C->B->A)
    // primer on 2D cross products:
    // http://allenchou.net/2013/07/cross-product-of-2d-vectors/
    function between(A, C, B) {
        var A_C = A.x * C.y - A.y * C.x, A_B = A.x * B.y - A.y * B.x, C_A = C.x * A.y - C.y * A.x, C_B = C.x * B.y - C.y * B.x;
        return A_C * A_B >= 0 && C_A * C_B >= 0;
    }

    var dir_vec = { x: e.gesture.deltaX, y: e.gesture.deltaY };

    if (between(n_vec, ne_vec, dir_vec)) {
        ne = true;
    } else if (between(ne_vec, se_vec, dir_vec)) {
        east = true;
    } else if (between(se_vec, s_vec, dir_vec)) {
        se = true;
    } else if (between(s_vec, sw_vec, dir_vec)) {
        sw = true;
    } else if (between(sw_vec, nw_vec, dir_vec)) {
        west = true;
    } else {
        nw = true;
    }

    function displace(set, direction) {
        return set.map(function (cell) {
            return { q: cell.q + direction.q, r: cell.r + direction.r };
        });
    }

    function checkCollision(newset, oldset) {
        return newset.map(function (cell, i) {
            // if cell is out of bounds, then, collision
            // if cell is not in original set and cell is not -1 then collision
            // if cell is not in original set and cell is -1 then no collision
            // if cell is in original set then no collsion
            var cellIsOutofBounds = grid.get(cell.q, cell.r).type == 0 /* OUT_OF_BOUNDS */;
            var cellInOldSet = oldset.some(function (c) {
                return c.q == cell.q && c.r == cell.r;
            });
            var isCollision = cellIsOutofBounds || (!cellInOldSet && grid.get(cell.q, cell.r).type != 1 /* EMPTY */);
            return isCollision;
        });
    }

    function move(from, to) {
        // cache all the from values before clearing them
        var fromVals = from.map(function (cell) {
            return grid.get(cell.q, cell.r);
        });
        from.forEach(function (cell) {
            grid.set(cell.q, cell.r, new Tile(1 /* EMPTY */, -1));
        });
        to.forEach(function (cell, i) {
            grid.set(cell.q, cell.r, new Tile(fromVals[i].type, fromVals[i].value));
        });

        for (var i = 0; i < from.length; i++) {
            var f = gridView.getDOMElement(from[i].q, from[i].r);
            var t = gridView.getDOMElement(to[i].q, to[i].r);

            gameState.disableMouse = true;

            var anim = f.animate(100, '>', 0).move(t.transform('x'), t.transform('y'));

            if (i == 0) {
                anim.after(function () {
                    gameState.disableMouse = false;
                    prune(to[0]);
                    update();
                    draw();
                    extendUI();
                });
            }
        }
    }

    prune(selected[0]);
    update();
    if (selected[0].type == 1 /* EMPTY */) {
        draw();
        extendUI();
        return;
    }

    var delta_vec = { q: 0, r: 0 };
    if (ne) {
        delta_vec.q = 1;
        delta_vec.r = -1;
    } else if (east) {
        delta_vec.q = 1;
        delta_vec.r = 0;
    } else if (se) {
        delta_vec.q = 0;
        delta_vec.r = 1;
    } else if (sw) {
        delta_vec.q = -1;
        delta_vec.r = 1;
    } else if (west) {
        delta_vec.q = -1;
        delta_vec.r = 0;
    } else {
        delta_vec.q = 0;
        delta_vec.r = -1;
    }

    var oldset = selected.map(Utils.deepCopy);
    var newset = oldset.map(Utils.deepCopy);
    while (checkCollision(newset, oldset).every(function (col) {
        return col == false;
    })) {
        oldset = newset.map(Utils.deepCopy); // oldset = newset (deep copy)
        newset = displace(oldset, delta_vec);
    }
    move(selected, oldset);
    selected = oldset; // shallow copy is fine

    gameState.selected = selected;
};

var init = function () {
    if (canvas == null)
        canvas = SVG('screen').size(720, 720);

    var gp = {
        gridw: 6,
        gridh: 6,
        maxVal: 5
    };

    var gs = {
        activeCells: gp.maxVal * (gp.maxVal + 1) / 2 - 1,
        disableMouse: false,
        selected: []
    };

    grid = new Model.Hex(Math.floor(gp.gridw / 2), new Tile(1 /* EMPTY */, 0), new Tile(0 /* OUT_OF_BOUNDS */, -1));

    gameParams = gp;
    gameState = gs;

    procGenGrid(grid, gp);
    gridView = View.fromModel(grid);

    draw();
    extendUI();

    Hammer(document.getElementById('screen'), { preventDefault: true }).on("dragend swipeend", onDrag);
};

// procedurally generate a playable grid
// the grid must:
// 1) have n of tiles labeled n.
// 2) have enough empty spaces to be tractable (and fun)
function procGenGrid(grid, gp) {
    // keep track of how many tiles there are for each number, indexed by number
    var count = [];
    for (var i = MIN_VAL; i <= gp.maxVal; i++) {
        count[i] = 0;
    }

    for (var i = 0; i < grid.getTileArray().length; i++) {
        if (grid.getFlat(i).type == 0 /* OUT_OF_BOUNDS */)
            continue;
        var val = Math.round(Math.random() * (gp.maxVal - MIN_VAL) + MIN_VAL);
        count[val]++;
        grid.setFlat(i, new Tile(2 /* REGULAR */, val));
    }

    for (i = 0; i < grid.getTileArray().length; i++) {
        if (Math.random() > 0.4) {
            if (count[grid.getFlat(i).value] > grid.getFlat(i).value) {
                count[grid.getFlat(i).value]--;
                grid.setFlat(i, new Tile(1 /* EMPTY */, -1));
            }
        }
    }

    for (i = MIN_VAL; i <= gp.maxVal; i++) {
        if (count[i] > i) {
            for (var j = 0; j < grid.getTileArray().length && count[i] > i; j++) {
                if (grid.getFlat(j).value == i) {
                    grid.setFlat(j, new Tile(1 /* EMPTY */, -1));
                    count[i]--;
                }
            }
        }
    }

    for (i = MIN_VAL; i <= gp.maxVal; i++) {
        if (count[i] < i) {
            while (count[i] < i) {
                var randIndex = Math.round(Math.random() * (grid.getTileArray().length - 1));
                if (grid.getFlat(randIndex).type == 1 /* EMPTY */) {
                    grid.setFlat(randIndex, new Tile(2 /* REGULAR */, i));
                    count[i]++;
                }
            }
        }
    }
}
