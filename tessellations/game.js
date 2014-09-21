/// <reference path="lib/chroma-js.d.ts" />
/// <reference path="lib/hammerjs.d.ts" />
/// <reference path="lib/svgjs.d.ts" />
/// <reference path="lib/utils.ts" />
/// <reference path="./model.ts" />
/// <reference path="./view.ts" />
var grid;
var gridView;
var gridw = 0, gridh = 0;
var gridRep = [];
var cellw = 120;
var cellh = 120;
var maxval = 4;
var canvas = null;
var activeCells = 0;
var level = 1;
var fastdebug = false;
var disableMouse = false;

var hover = [];
var selected = [];

// weird to do this, but I want to instantiate this object array/hash with TileType vals
// and I can't do that in the constructor like above, because JavaScript doesn't like periods
// in its initializers.
var enabled = (function () {
    this[5 /* LAVA */] = false;
    return this;
})();

var colorizer = (function () {
    var colorizer = { scale: null, fromValue: null, highlightFromValue: null, borderFromValue: null };

    colorizer.scale = {};
    colorizer.scale[2 /* REGULAR */] = chroma.scale(['#4BF920', '#1DE5A2', '#48CC20', '#18BC49', '#0DAD6D']);
    colorizer.scale[5 /* LAVA */] = chroma.scale(['#AE5750', '#F96541', '#FF7939']);
    colorizer.scale[4 /* DEACTIVATED */] = chroma.scale(['#64585A', '#64585A']);

    colorizer.fromValue = function (t, v) {
        if (v < 0)
            return chroma('white').hex();
        else
            return this.scale[t](v / 9).hex();
    };

    colorizer.highlightFromValue = function (t, v) {
        if (v < 0)
            return chroma('white').hex();
        else
            return this.scale[t](v / 9).brighter().hex();
    };

    colorizer.borderFromValue = function (t, v) {
        if (v < 0)
            return chroma('white').hex();
        else
            return this.scale[t](v / 9).darker().hex();
    };

    return colorizer;
})();

// translation of practical flood fill implementation as described on
// http://en.wikipedia.org/wiki/Flood_fill
var floodAcquire = function (start, tile) {
    var cluster = [];
    var marked = { get: null, set: null };
    marked.get = function (x, y) {
        return this[x + y * gridw] === undefined ? false : this[x + y * gridw];
    };
    marked.set = function (x, y) {
        this[x + y * gridw] = true;
    };
    var Q = [];
    if (grid.get(new CartesianCoords(start.x, start.y)) != tile)
        return [];
    Q.push(start);
    while (Q.length > 0) {
        var n = Q.shift();
        var c = new CartesianCoords(n.x, n.y);
        if (grid.get(c).value == tile.value && grid.get(c).type == tile.type && !marked.get(n.x, n.y)) {
            var w = { x: n.x, y: n.y };
            var e = { x: n.x, y: n.y };

            while (grid.get(new CartesianCoords(w.x - 1, w.y)).value == tile.value && grid.get(new CartesianCoords(w.x - 1, w.y)).type == tile.type) {
                w.x--;
            }

            while (grid.get(new CartesianCoords(e.x + 1, e.y)).value == tile.value && grid.get(new CartesianCoords(e.x + 1, e.y)).type == tile.type) {
                e.x++;
            }

            for (var x = w.x; x < e.x + 1; x++) {
                var nn = { x: x, y: n.y };
                marked.set(nn.x, nn.y);
                cluster.push(nn);
                var north = { x: nn.x, y: nn.y - 1 };
                var south = { x: nn.x, y: nn.y + 1 };
                if (grid.get(new CartesianCoords(north.x, north.y)).value == tile.value && grid.get(new CartesianCoords(north.x, north.y)).type == tile.type)
                    Q.push(north);
                if (grid.get(new CartesianCoords(south.x, south.y)).value == tile.value && grid.get(new CartesianCoords(south.x, south.y)).type == tile.type)
                    Q.push(south);
            }
        }
    }
    return cluster;
};

var draw = function () {
    canvas.clear();

    // draw cells
    gridRep = [];
    for (var y = 0; y < gridh; y++) {
        for (var x = 0; x < gridw; x++) {
            var model = { x: x, y: y, i: x + y * gridw };
            var e = grid.get(new CartesianCoords(x, y));

            var cell = canvas.group().transform({ x: x * cellw, y: y * cellh });

            var rect = cell.rect(cellw, cellh);

            rect.node.model = model; // so the DOM can refer to this to get its x/y/i/whatever

            rect.attr({
                'fill': colorizer.fromValue(e.type, e.value),
                'fill-opacity': e.type == 1 /* EMPTY */ ? 0 : 1,
                'id': 'cell' + model.i }).mouseover(function () {
                if (disableMouse)
                    return;

                // 'this' refers to a wrapper provided by SVGjs, so we have to go down to node to get the model
                if (grid.get(new CartesianCoords(this.node.model.x, this.node.model.y)).type == 4 /* DEACTIVATED */)
                    return;
                var target = { x: this.node.model.x, y: this.node.model.y };
                hover = floodAcquire(target, grid.get(new CartesianCoords(target.x, target.y)));
                hover.forEach(function (t) {
                    var i = t.x + t.y * gridw;
                    gridRep[i].rect.attr({ fill: colorizer.highlightFromValue(grid.getFlat(i).type, grid.getFlat(i).value) });
                });
            }).mouseout(function () {
                if (disableMouse)
                    return;
                var target = { x: this.node.model.x, y: this.node.model.y };
                hover = floodAcquire(target, grid.get(new CartesianCoords(target.x, target.y)));
                hover.forEach(function (t) {
                    var i = t.x + t.y * gridw;
                    gridRep[i].rect.attr({ fill: colorizer.fromValue(grid.getFlat(i).type, grid.getFlat(i).value) });
                });
            });

            var hammer = Hammer(rect.node, { preventDefault: true });
            hammer.on("dragstart swipestart", function (e) {
                // 'this' refers to the DOM node directly here
                if (grid.get(new CartesianCoords(this.model.x, this.model.y)).type != 4 /* DEACTIVATED */) {
                    var target = { x: this.model.x, y: this.model.y };
                    selected = floodAcquire(target, grid.get(new CartesianCoords(target.x, target.y)));
                } else {
                    selected = [];
                }
            });

            var text = cell.plain(e.type != 1 /* EMPTY */ ? e.value.toString() : "").fill({ color: '#ffffff' }).transform({ x: cellw / 2, y: cellh / 2 });

            gridRep.push({ cell: cell, rect: rect, text: text });
        }
    }
};

var update = function () {
    gridRep.forEach(function (rep, i) {
        if (!grid.getFlat(i)) {
            console.log("tile " + i + " is undefined");
            return;
        }
        rep.text.plain(grid.getFlat(i).type != 1 /* EMPTY */ ? grid.getFlat(i).value.toString() : "");
        rep.rect.attr({ fill: colorizer.fromValue(grid.getFlat(i).type, grid.getFlat(i).value) });
    });
};

var advance = function () {
    level++;

    if (fastdebug) {
        cellw = Math.round(cellw * 0.5);
        cellh = Math.round(cellh * 0.5);
    } else {
        cellw = Math.round(cellw * 0.8);
        cellh = Math.round(cellh * 0.8);
    }

    maxval++;

    if (level > 3 || fastdebug) {
        enabled[5 /* LAVA */] = true;
    }

    init();
};

var prune = function (start) {
    // see if we should delete this cell and surrounding cells
    var targets = floodAcquire(start, grid.get(new CartesianCoords(start.x, start.y)));
    if (targets.length == grid.get(new CartesianCoords(start.x, start.y)).value) {
        if (grid.get(new CartesianCoords(start.x, start.y)).type == 4 /* DEACTIVATED */)
            return;
        targets.forEach(function (cell) {
            if (grid.get(new CartesianCoords(cell.x, cell.y)).type == 2 /* REGULAR */) {
                grid.set(new CartesianCoords(cell.x, cell.y), new Tile(1 /* EMPTY */, -1));
                activeCells--;
            } else if (grid.get(new CartesianCoords(cell.x, cell.y)).type == 5 /* LAVA */) {
                grid.set(new CartesianCoords(cell.x, cell.y), new Tile(4 /* DEACTIVATED */, grid.get(new CartesianCoords(cell.x, cell.y)).value));
            }

            console.log(activeCells);
        });
    }

    // this is a bad cross-cutting dependency
    if (activeCells == 0) {
        alert("great job, you won!");
        advance();
    }
};

var pollDrag = function (e) {
    if (selected == null || selected.length == 0) {
        console.log("nothing selected");
        return;
    }

    var up = false, down = false, left = false, right = false;

    if (Math.abs(e.gesture.deltaY) > Math.abs(e.gesture.deltaX)) {
        up = e.gesture.deltaY < 0;
        down = !up;
    } else {
        left = e.gesture.deltaX < 0;
        right = !left;
    }

    function displace(set, direction) {
        return set.map(function (cell) {
            return { x: cell.x + direction.x, y: cell.y + direction.y };
        });
    }

    function checkCollision(newset, oldset) {
        return newset.map(function (cell, i) {
            // if cell is out of bounds, then, collision
            // if cell is not in original set and cell is not -1 then collision
            // if cell is not in original set and cell is -1 then no collision
            // if cell is in original set then no collsion
            var cellIsOutofBounds = grid.get(new CartesianCoords(cell.x, cell.y)).type == 0 /* OUT_OF_BOUNDS */;
            var cellInOldSet = oldset.some(function (c) {
                return c.x == cell.x && c.y == cell.y;
            });
            var isCollision = cellIsOutofBounds || (!cellInOldSet && grid.get(new CartesianCoords(cell.x, cell.y)).type != 1 /* EMPTY */);
            return isCollision;
        });
    }

    function move(from, to) {
        // cache all the from values before clearing them
        var fromVals = from.map(function (cell) {
            return grid.get(new CartesianCoords(cell.x, cell.y));
        });
        from.forEach(function (cell) {
            grid.set(new CartesianCoords(cell.x, cell.y), new Tile(1 /* EMPTY */, -1));
        });
        to.forEach(function (cell, i) {
            grid.set(new CartesianCoords(cell.x, cell.y), new Tile(fromVals[i].type, fromVals[i].value));
        });

        for (var i = 0; i < from.length; i++) {
            var f = gridRep[from[i].x + gridw * from[i].y].cell;
            var t = gridRep[to[i].x + gridw * to[i].y].cell;

            disableMouse = true;

            var anim = f.animate(100, '>', 0).move(t.transform('x'), t.transform('y'));

            if (i == 0) {
                anim.after(function () {
                    disableMouse = false;
                    prune(to[0]);
                    update();
                    draw();
                });
            }
        }
    }

    prune(selected[0]);
    if (selected[0].type == 1 /* EMPTY */) {
        update();
        draw();
        return;
    }

    if (up) {
        var oldset = selected.map(Utils.deepCopy);
        var newset = oldset.map(Utils.deepCopy);
        while (checkCollision(newset, oldset).every(function (col) {
            return col == false;
        })) {
            oldset = newset.map(Utils.deepCopy); // oldset = newset (deep copy)
            newset = displace(oldset, { x: 0, y: -1 });
        }
        move(selected, oldset);
        selected = oldset; // shallow copy is fine
    } else if (down) {
        var oldset = selected.map(Utils.deepCopy);
        var newset = oldset.map(Utils.deepCopy);
        while (checkCollision(newset, oldset).every(function (col) {
            return col == false;
        })) {
            oldset = newset.map(Utils.deepCopy); // oldset = newset (deep copy)
            newset = displace(oldset, { x: 0, y: 1 });
        }
        move(selected, oldset);
        selected = oldset; // shallow copy is fine
    }

    if (left) {
        var oldset = selected.map(Utils.deepCopy);
        var newset = oldset.map(Utils.deepCopy);
        while (checkCollision(newset, oldset).every(function (col) {
            return col == false;
        })) {
            oldset = newset.map(Utils.deepCopy);
            newset = displace(oldset, { x: -1, y: 0 });
        }
        move(selected, oldset);
        selected = oldset; // shallow copy is fine
    } else if (right) {
        var oldset = selected.map(Utils.deepCopy);
        var newset = oldset.map(Utils.deepCopy);
        while (checkCollision(newset, oldset).every(function (col) {
            return col == false;
        })) {
            oldset = newset.map(Utils.deepCopy);
            newset = displace(oldset, { x: 1, y: 0 });
        }
        move(selected, oldset);
        selected = oldset; // shallow copy is fine
    }
};

var init = function () {
    if (canvas == null)
        canvas = SVG('screen').size(720, 720);

    gridw = Math.floor(canvas.width() / cellw);
    gridh = Math.floor(canvas.height() / cellh);

    console.log("new grid size is " + gridw + "x" + gridh);

    var tiles = [];

    grid = new Model.Square(gridw, gridh, new Tile(1 /* EMPTY */, 0), new Tile(0 /* OUT_OF_BOUNDS */, -1));

    for (var i = 0; i < gridw * gridh; i++) {
        var val = Math.round(Math.random() * (maxval - 2) + 2);
        grid.setFlat(i, new Tile(2 /* REGULAR */, val));
        if (tiles[val] == null) {
            tiles[val] = 0;
        }
        tiles[val]++;
    }

    for (i = 0; i < gridw * gridh; i++) {
        if (Math.random() > 0.4) {
            if (tiles[grid.getFlat(i).value] > grid.getFlat(i).value) {
                tiles[grid.getFlat(i).value]--;
                grid.setFlat(i, new Tile(1 /* EMPTY */, -1));
            }
        }
    }

    for (i = 2; i <= maxval; i++) {
        if (tiles[i] > i) {
            var count = tiles[i];
            for (var j = 0; j < gridw * gridh && tiles[i] > i; j++) {
                if (grid.getFlat(j).value == i) {
                    grid.setFlat(j, new Tile(1 /* EMPTY */, -1));
                    tiles[i]--;
                }
            }
        }
    }

    for (i = 2; i <= maxval; i++) {
        if (tiles[i] < i) {
            var count = tiles[i];
            while (tiles[i] < i) {
                var randIndex = Math.round(Math.random() * (gridw * gridh - 1));
                if (grid.getFlat(randIndex).type == 1 /* EMPTY */) {
                    grid.setFlat(randIndex, new Tile(2 /* REGULAR */, i));
                    tiles[i]++;
                }
            }
        }
    }

    // count active cells
    activeCells = 0;
    for (i = 2; i <= maxval; i++) {
        if (tiles[i]) {
            activeCells += tiles[i];
        } else {
            console.log("bad state:" + i + " is null");
        }
    }

    if (enabled[5 /* LAVA */]) {
        tiles = [];
        for (i = 0; i < gridw * gridh; i++) {
            if (grid.getFlat(i).type == 1 /* EMPTY */) {
                if (Math.random() > 0.4) {
                    var val = Math.round(Math.random() * (maxval - 4) + 2);
                    grid.setFlat(i, new Tile(5 /* LAVA */, val));
                    if (tiles[val] == null) {
                        tiles[val] = 0;
                    }
                    tiles[val]++;
                }
            }
        }

        for (i = 0; i < gridw * gridh; i++) {
            if (Math.random() > 0.4) {
                if (grid.getFlat(i).type == 5 /* LAVA */ && tiles[grid.getFlat(i).value] > grid.getFlat(i).value) {
                    tiles[grid.getFlat(i).value]--;
                    grid.setFlat(i, new Tile(1 /* EMPTY */, -1));
                }
            }
        }

        for (i = 2; i <= maxval; i++) {
            if (tiles[i] > i) {
                var count = tiles[i];
                for (var j = 0; j < gridw * gridh && tiles[i] > i; j++) {
                    if (grid.getFlat(j).type == 5 /* LAVA */ && grid.getFlat(j).value == i) {
                        grid.setFlat(j, new Tile(1 /* EMPTY */, -1));
                        tiles[i]--;
                    }
                }
            }
        }

        for (i = 2; i <= maxval; i++) {
            if (tiles[i] < i) {
                var count = tiles[i];
                while (tiles[i] < i) {
                    var randIndex = Math.round(Math.random() * (gridw * gridh - 1));
                    if (grid.getFlat(randIndex).type == 1 /* EMPTY */) {
                        grid.setFlat(randIndex, new Tile(5 /* LAVA */, i));
                        tiles[i]++;
                    }
                }
            }
        }
    }

    gridView = View.fromModel(grid);

    draw();

    Hammer(document.getElementById('screen'), { preventDefault: true }).on("dragend swipeend", pollDrag);
};
