var Utils = (function () {
    function Utils() {
    }
    Utils.deepCopy = function (src) {
        var dst = Object.create(Object.getPrototypeOf(src));
        var keys = Object.keys(src);
        keys.map(function (key) {
            return dst[key] = src[key];
        });
        return dst;
    };

    // http://stackoverflow.com/a/11582513
    Utils.getURLParameter = function (key) {
        return decodeURIComponent((new RegExp('[?|&]' + key + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search) || [, ""])[1].replace(/\+/g, '%20')) || null;
    };

    Utils.getOffset = function (elem) {
        var left = 0;
        var top = 0;
        while (true) {
            left += elem.offsetLeft;
            top += elem.offsetTop;
            if (elem.offsetParent === null) {
                break;
            }
            elem = elem.offsetParent;
        }
        return { top: top, left: left };
    };

    Utils.EPSILON = 0.00001;
    return Utils;
})();
/// <reference path="lib/utils.ts" />
var TileType;
(function (TileType) {
    TileType[TileType["OUT_OF_BOUNDS"] = 0] = "OUT_OF_BOUNDS";
    TileType[TileType["EMPTY"] = 1] = "EMPTY";
    TileType[TileType["REGULAR"] = 2] = "REGULAR";
    TileType[TileType["CONCRETE"] = 3] = "CONCRETE";
    TileType[TileType["DEACTIVATED"] = 4] = "DEACTIVATED";
    TileType[TileType["LAVA"] = 5] = "LAVA";
})(TileType || (TileType = {}));
;


var CardinalDirection;
(function (CardinalDirection) {
    CardinalDirection[CardinalDirection["NORTH"] = 0] = "NORTH";
    CardinalDirection[CardinalDirection["EAST"] = 1] = "EAST";
    CardinalDirection[CardinalDirection["SOUTH"] = 2] = "SOUTH";
    CardinalDirection[CardinalDirection["WEST"] = 3] = "WEST";
})(CardinalDirection || (CardinalDirection = {}));

var CartesianCoords = (function () {
    function CartesianCoords(x, y) {
        this.x = x;
        this.y = y;
    }
    CartesianCoords.prototype.equals = function (that) {
        return that != null && this.x == that.x && this.y == that.y;
    };
    CartesianCoords.prototype.displace = function (arg) {
        if (arg instanceof CartesianCoords) {
            return new CartesianCoords(this.x + arg.x, this.y + arg.y);
        } else {
            var dir = arg;
            switch (dir) {
                case 0 /* NORTH */:
                    return new CartesianCoords(this.x, this.y - 1);
                    break;
                case 2 /* SOUTH */:
                    return new CartesianCoords(this.x, this.y + 1);
                    break;
                case 3 /* WEST */:
                    return new CartesianCoords(this.x - 1, this.y);
                    break;
                case 1 /* EAST */:
                    return new CartesianCoords(this.x + 1, this.y);
                    break;
            }
        }
    };
    CartesianCoords.displace = function (coords, direction) {
        return new CartesianCoords(coords.x + direction.x, coords.y + direction.y);
    };
    CartesianCoords.prototype.toString = function () {
        return "( " + this.x + ", " + this.y + " )";
    };
    return CartesianCoords;
})();

// okay, there's probably a better name for this
var CartesianBounds = (function () {
    function CartesianBounds(n, e, s, w) {
        this.n = n != undefined ? n : -1;
        this.e = e != undefined ? e : -1;
        this.s = s != undefined ? s : -1;
        this.w = w != undefined ? s : -1;
    }
    return CartesianBounds;
})();

var Tile = (function () {
    function Tile(t, v) {
        this.type = t;
        this.value = v;
        this.bounds = new CartesianBounds();
    }
    Tile.prototype.toString = function () {
        return this.value.toString();
    };
    return Tile;
})();

var Model;
(function (Model) {
    var Square = (function () {
        function Square(gridw, gridh, DefaultTile, OutOfBoundsTile) {
            this.outOfBoundsTile = OutOfBoundsTile;
            this.gridw = gridw;
            this.gridh = gridh;
            this.size = gridw * gridh;
            this.grid = [];
            for (var i = 0; i < this.gridw * this.gridh; i++) {
                this.grid.push(DefaultTile);
            }
        }
        Square.prototype.toFlat = function (x, y) {
            return x + y * this.gridw;
        };

        Square.prototype.get = function (c) {
            return c.y >= 0 && c.x >= 0 && c.x < this.gridw && c.y < this.gridh ? this.grid[c.x + c.y * this.gridw] : this.outOfBoundsTile;
        };

        Square.prototype.getFlat = function (i) {
            return i >= 0 && i < this.gridw * this.gridh ? this.grid[i] : this.outOfBoundsTile;
        };

        Square.prototype.set = function (c, tile) {
            if (c.y >= 0 && c.x >= 0 && c.x < this.gridw && c.y < this.gridh) {
                this.grid[c.x + c.y * this.gridw] = tile;
            }
        };

        Square.prototype.setFlat = function (i, tile) {
            if (i >= 0 && i < this.gridw * this.gridh) {
                this.grid[i] = tile;
            }
        };

        Square.prototype.getTileArray = function () {
            return this.grid;
        };

        Square.prototype.floodAcquire = function (start) {
            var cluster = [];
            var marked = { get: null, set: null };
            marked.get = function (key) {
                return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
            };
            marked.set = function (key) {
                this[JSON.stringify(key)] = true;
            };
            var Q = [];
            var tile = this.get(start);
            if (this.get(new CartesianCoords(start.x, start.y)) != tile)
                return [];
            Q.push(start);
            while (Q.length > 0) {
                var n = Q.shift();
                if (this.get(n).value == tile.value && this.get(n).type == tile.type && !marked.get(n)) {
                    var w = new CartesianCoords(n.x, n.y);
                    var e = new CartesianCoords(n.x, n.y);

                    while (this.get(new CartesianCoords(w.x - 1, w.y)).value == tile.value && this.get(new CartesianCoords(w.x - 1, w.y)).type == tile.type) {
                        w.x--;
                    }

                    while (this.get(new CartesianCoords(e.x + 1, e.y)).value == tile.value && this.get(new CartesianCoords(e.x + 1, e.y)).type == tile.type) {
                        e.x++;
                    }

                    for (var x = w.x; x < e.x + 1; x++) {
                        var nn = new CartesianCoords(x, n.y);
                        marked.set(nn);
                        cluster.push(nn);
                        var north = new CartesianCoords(nn.x, nn.y - 1);
                        var south = new CartesianCoords(nn.x, nn.y + 1);
                        if (this.get(north).value == tile.value && this.get(north).type == tile.type)
                            Q.push(north);
                        if (this.get(south).value == tile.value && this.get(south).type == tile.type)
                            Q.push(south);
                    }
                }
            }
            return cluster;
        };

        Square.prototype.checkCollision = function (from, future) {
            var _this = this;
            return future.map(function (cell, i) {
                // if cell is in original set then no collsion
                // if cell is out of bounds, then, collision
                var ignoreCollision = from.some(function (c) {
                    return c != undefined && c.x == cell.x && c.y == cell.y;
                });
                var cellIsOutofBounds = _this.get(cell).type == 0 /* OUT_OF_BOUNDS */;
                var isCollision = cellIsOutofBounds || (!ignoreCollision && _this.get(cell).type != 1 /* EMPTY */);
                return isCollision;
            });
        };

        Square.prototype.recomputeAllBounds = function () {
            var _this = this;
            this.grid.forEach(function (t, i) {
                if (t.type != 1 /* EMPTY */)
                    t.bounds = _this.computeBounds(new CartesianCoords(i % _this.gridw, Math.floor(i / _this.gridw)));
            });
        };

        Square.prototype.getTileBoundInDirection = function (c, dir) {
            var b = this.get(c).bounds;
            switch (dir) {
                case 0 /* NORTH */:
                    return new CartesianCoords(c.x, b.n);
                case 2 /* SOUTH */:
                    return new CartesianCoords(c.x, b.s);
                case 3 /* WEST */:
                    return new CartesianCoords(b.w, c.y);
                case 1 /* EAST */:
                    return new CartesianCoords(b.e, c.y);
            }
        };

        Square.prototype.computeBounds = function (c) {
            var group = this.floodAcquire(c);

            // my index is important; find it
            var i = -1;
            for (var i = 0; i < group.length; i++) {
                if (JSON.stringify(group[i]) == JSON.stringify(c)) {
                    break;
                }
            }

            var dest;
            var future;

            function resetDestFuture() {
                dest = group.map(Utils.deepCopy);
                future = dest.map(Utils.deepCopy);
            }

            resetDestFuture();

            var bounds = new CartesianBounds();

            var n = new CartesianCoords(0, -1);
            var s = new CartesianCoords(0, 1);
            var w = new CartesianCoords(-1, 0);
            var e = new CartesianCoords(1, 0);

            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(n);
                });
            }

            bounds.n = dest[i].y;
            resetDestFuture();

            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(s);
                });
            }

            bounds.s = dest[i].y;
            resetDestFuture();

            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(w);
                });
            }

            bounds.w = dest[i].x;
            resetDestFuture();

            while (this.checkCollision(group, future).every(function (col) {
                return col == false;
            })) {
                dest = future.map(Utils.deepCopy);
                future = dest.map(function (c) {
                    return c.displace(e);
                });
            }

            bounds.e = dest[i].x;
            resetDestFuture();

            return bounds;
        };

        Square.prototype.debugPrint = function () {
            var _this = this;
            var str = "";
            this.grid.forEach(function (t, i) {
                if (i % _this.gridw == 0) {
                    str += "\n";
                }
                str += (t.type != 1 /* EMPTY */ ? t.value.toString() : " ") + " ";
            });
            return str;
        };
        return Square;
    })();
    Model.Square = Square;
})(Model || (Model = {}));
/// <reference path="lib/chroma-js.d.ts" />
/// <reference path="lib/svgjs.d.ts" />
/// <reference path="lib/fabricjs.d.ts" />
/// <reference path="lib/tsm-0.7.d.ts" />
/// <reference path="./model.ts" />
var DrawBackend;
(function (DrawBackend) {
    DrawBackend[DrawBackend["CANVAS"] = 0] = "CANVAS";
    DrawBackend[DrawBackend["SVG"] = 1] = "SVG";
})(DrawBackend || (DrawBackend = {}));

var SquareDim = (function () {
    function SquareDim() {
    }
    return SquareDim;
})();

var Colorizer = (function () {
    function Colorizer() {
        this.highlightFromTile = function (t) {
            if (t.type == 1 /* EMPTY */) {
                return this.scale[t.type](t.value / 9).hex();
            } else {
                return this.scale[t.type](t.value / 9).brighter().hex();
            }
        };
        this.shadowFromTile = function (t) {
            if (t.type == 1 /* EMPTY */) {
                return this.scale[t.type](t.value / 9).hex();
            } else {
                return this.scale[t.type](t.value / 9).darken(10).hex();
            }
        };
        this.foregroundFromColor = function (c) {
            return chroma(c).lab()[0] > 70 ? '#000' : '#fff';
        };
        this.scale = {};
        this.scale[1 /* EMPTY */] = chroma.scale(['#D7FAF3', '#F3F4E5', '#FFFFFF']);
        this.scale[2 /* REGULAR */] = chroma.scale(['#FFF', 'FFABAB', '#FBCA04', '#1DE5A2', '#3692B9', '#615258', '#EB6420', '#C8FF00', '#46727F', '#1D1163']);
        this.scale[5 /* LAVA */] = chroma.scale(['#AE5750', '#F96541', '#FF7939']);
        this.scale[4 /* DEACTIVATED */] = chroma.scale(['#64585A', '#64585A']);
    }
    Colorizer.prototype.fromTile = function (t) {
        return this.scale[t.type](t.value / 9).hex();
    };
    return Colorizer;
})();

// quick-n-dirty
var Vec2 = (function () {
    function Vec2(x, y) {
        this.x = x;
        this.y = y;
    }
    Vec2.prototype.toString = function () {
        return this.x + "," + this.y;
    };
    return Vec2;
})();

var View;
(function (View) {
    function buildCanvas(w, h, backend) {
        switch (backend) {
            case 0 /* CANVAS */:
                var canvas = document.createElement("canvas");
                canvas.setAttribute("id", "screen-canvas");
                canvas.setAttribute("width", "640");
                canvas.setAttribute("height", "650");
                document.getElementById("screen").appendChild(canvas);
                return new fabric.Canvas("screen-canvas", { selection: false });
                break;
            case 1 /* SVG */:
                // need to fix svgjs.d.ts
                return SVG('screen').size(w, h).attr("preserveAspectRatio", "none");
                break;
        }
    }
    View.buildCanvas = buildCanvas;

    function build(grid, backend) {
        return new SquareView(grid, backend);
    }
    View.build = build;

    var SquareView = (function () {
        function SquareView(square, backend) {
            this.drawBackend = backend;
            this.model = square;
            this.colorizer = new Colorizer();
        }
        SquareView.prototype.getCellSize = function (canvas) {
            var cellw = 0, cellh = 0;
            switch (this.drawBackend) {
                case 0 /* CANVAS */:
                    cellw = canvas.getWidth() / this.model.gridw, cellh = canvas.getWidth() / this.model.gridh;
                    break;
                case 1 /* SVG */:
                    cellw = canvas.width() / this.model.gridw, cellh = canvas.width() / this.model.gridh;
                    break;
            }
            var d = { w: cellw, h: cellh };
            return d;
        };

        SquareView.prototype.getGridCoordsFromScreenPos = function (canvas, pos) {
            var size = this.getCellSize(canvas);
            var cellw = size.w, cellh = size.h;
            return new CartesianCoords(Math.floor(pos.x / cellw), Math.floor(pos.y / cellh));
        };

        SquareView.prototype.drawTile = function (canvas, x, y, e, dragOffset) {
            var cellw = this.getCellSize(canvas).w, cellh = this.getCellSize(canvas).h;

            var xOffset = cellw / 2;
            var yOffset = cellh / 2;

            var cell;

            if (this.drawBackend == 1 /* SVG */) {
                cell = canvas.group().transform({ x: x * cellw + xOffset + dragOffset.x, y: y * cellh + yOffset + dragOffset.y });

                var pts = [new Vec2(-cellw / 2, -cellh / 2), new Vec2(cellw / 2, -cellh / 2), new Vec2(cellw / 2, cellh / 2), new Vec2(-cellw / 2, cellh / 2)];
                var ptstr = pts.reduce(function (p1, p2, i, v) {
                    return p1.toString() + " " + p2.toString();
                }, "");

                var rect = cell.polygon(ptstr);

                rect.attr({
                    'fill': this.colorizer.fromTile(e),
                    'fill-opacity': e.type == 1 /* EMPTY */ ? 0 : 1 });

                var text = cell.plain(e.type != 1 /* EMPTY */ ? e.value.toString() : "");

                text.attr({
                    'fill': this.colorizer.foregroundFromColor(this.colorizer.fromTile(e)),
                    'font-size': cellw / 4 }).transform({ x: -text.attr('font-size') / 4, y: text.attr('font-size') / 4 });

                // cache UI hooks
                cell.coords = new CartesianCoords(x, y);
                cell.rect = rect;
                cell.text = text;
                cell.e = e;
                cell.cannonicalTransform = { x: x * cellw + xOffset, y: y * cellh + yOffset };
                cell.dragOffset = dragOffset;
            } else if (this.drawBackend == 0 /* CANVAS */) {
                var rect = new fabric.Rect({
                    width: cellw,
                    height: cellh,
                    fill: this.colorizer.fromTile(e),
                    originX: 'center',
                    originY: 'center'
                });
                var text = new fabric.Text(e.type != 1 /* EMPTY */ ? e.value.toString() : "", {
                    originX: 'center',
                    originY: 'center',
                    fill: this.colorizer.foregroundFromColor(this.colorizer.fromTile(e)),
                    fontSize: cellw / 4,
                    fontFamily: 'sans-serif',
                    hoverCursor: 'default'
                });
                cell = new fabric.Group([rect, text], {
                    left: x * cellw + xOffset + dragOffset.x - cellw / 2,
                    top: y * cellh + yOffset + dragOffset.y - cellh / 2
                });

                // cache UI hooks
                cell.coords = new CartesianCoords(x, y);
                cell.rect = rect;
                cell.text = text;
                cell.e = e;
                cell.cannonicalLeft = cell.left;
                cell.cannonicalTop = cell.top;
                cell.dragOffset = dragOffset;
                if (e.type != 1 /* EMPTY */)
                    canvas.add(cell);
            }
            return cell;
        };

        SquareView.prototype.updateTimerBar = function (canvas, frac, color) {
            if (this.drawBackend == 1 /* SVG */) {
                if (canvas == null) {
                    if (this.timerBar != null) {
                        this.timerBar.remove();
                    }
                    return;
                }
                var oldw = 0;
                ;
                if (this.timerBar != null) {
                    oldw = this.timerBar.width();
                    this.timerBar.remove();
                }
                if (color == null) {
                    color = this.colorizer.scale[2 /* REGULAR */](1 / 9).hex();
                }
                var barHeight = 10;
                this.timerBar = canvas.rect(oldw, barHeight).radius(0);
                this.timerBar.attr({ 'fill': color }).transform({ x: 0, y: canvas.height() - barHeight }).animate(100, '>', 0).attr({ 'width': frac * canvas.width() });
            }
        };

        SquareView.prototype.resetView = function (canvas) {
            console.log("resetView");
            if (this.model == null)
                return;
            if (this.cells == null)
                this.cells = [];

            if (this.drawBackend == 0 /* CANVAS */)
                canvas.clear();

            for (var y = 0; y < this.model.gridh; y++) {
                for (var x = 0; x < this.model.gridw; x++) {
                    var e = this.model.get(new CartesianCoords(x, y));
                    var offs = new TSM.vec2([0, 0]);
                    if (this.cells[this.model.toFlat(x, y)] != null) {
                        // offs = this.cells[ this.model.toFlat( x, y ) ].dragOffset;
                        if (this.drawBackend == 1 /* SVG */) {
                            this.cells[this.model.toFlat(x, y)].clear();
                        } else {
                            this.cells[this.model.toFlat(x, y)].remove();
                        }
                    }
                    this.cells[this.model.toFlat(x, y)] = this.drawTile(canvas, x, y, e, offs);
                }
            }
        };

        SquareView.prototype.getViewElements = function () {
            return this.cells;
        };

        SquareView.prototype.getViewElement = function (c) {
            return c.y >= 0 && c.x >= 0 && c.x < this.model.gridw && c.y < this.model.gridh ? this.cells[this.model.toFlat(c.x, c.y)] : null;
        };

        SquareView.prototype.debugPrint = function () {
            var _this = this;
            var str = "";
            this.cells.forEach(function (c, i) {
                if (i % _this.model.gridw == 0) {
                    str += "\n";
                }
                str += (c.e.type != 1 /* EMPTY */ ? c.e.value.toString() : " ") + " ";
            });
            return str;
        };

        SquareView.prototype.slide = function (canvas, from, to, doneCallback) {
            var _this = this;
            if (this.drawBackend == 1 /* SVG */) {
                var e1 = this.getViewElement(from[0]);
                var e2 = this.getViewElement(to[0]);

                if (e1 == null || e2 == null)
                    debugger;
                var d = Math.abs(e2.transform('x') - e1.transform('x')) + Math.abs(e2.transform('y') - e1.transform('y'));
                var speed = 2;

                canvas.animate(d / speed, '>', 0).during(function (t) {
                    from.forEach(function (f, i) {
                        var fromElement = _this.getViewElement(f);
                        var toElement = _this.getViewElement(to[i]);

                        var x = fromElement.cannonicalTransform.x + fromElement.dragOffset.x + (toElement.cannonicalTransform.x - fromElement.cannonicalTransform.x - fromElement.dragOffset.x) * SVG.easing.quadOut(t);
                        var y = fromElement.cannonicalTransform.y + fromElement.dragOffset.y + (toElement.cannonicalTransform.y - fromElement.cannonicalTransform.y - fromElement.dragOffset.y) * SVG.easing.quadOut(t);

                        fromElement.translate(x, y);
                    });
                }).after(function () {
                    from.forEach(function (f, i) {
                        var fromElement = _this.getViewElement(f);
                        var toElement = _this.getViewElement(to[i]);
                        fromElement.dragOffset.x = 0;
                        fromElement.dragOffset.y = 0;
                    });
                    doneCallback();
                });
            } else if (this.drawBackend == 0 /* CANVAS */) {
                // THIS IS WRONG
                from.forEach(function (f, i) {
                    var fromElement = _this.getViewElement(f);
                    var toElement = _this.getViewElement(to[i]);
                    fromElement.left = toElement.left;
                    fromElement.top = toElement.top;
                });
                canvas.renderAll();
            }
        };

        SquareView.prototype.crush = function (canvas, c, dir, doneCallback) {
            var cell = this.getViewElement(c);
            var cellh = this.getCellSize(canvas).h;
            var horiz = dir == 3 /* WEST */ || dir == 1 /* EAST */;
            var vert = !horiz;
            if (this.drawBackend == 1 /* SVG */) {
                cell.animate(100, '<', 0).scale(horiz ? 0 : 1, vert ? 0 : 1).move(horiz ? (cell.cannonicalTransform.x + (dir == 3 /* WEST */ ? -1 : 1) * cellh / 2) : cell.cannonicalTransform.x, vert ? (cell.cannonicalTransform.y + (dir == 0 /* NORTH */ ? -1 : 1) * cellh / 2) : cell.cannonicalTransform.y).after(function () {
                    doneCallback();
                });
            }
        };

        SquareView.prototype.pop = function (canvas, cs, doneCallback) {
            var _this = this;
            canvas.animate(200, '>', 0).during(function (t) {
                cs.forEach(function (c, i) {
                    var element = _this.getViewElement(c);
                    if (_this.drawBackend == 1 /* SVG */) {
                        // scale is [0-1], so no need to save startScaleX/startScaleY here
                        var scaleX = 1 - SVG.easing.quadOut(t);
                        var scaleY = 1 - SVG.easing.quadOut(t);
                        element.scale(scaleX, scaleY);
                    } else if (_this.drawBackend == 0 /* CANVAS */) {
                        // no idea
                    }
                });
            }).after(function () {
                doneCallback();
            });
        };

        SquareView.prototype.materialize = function (canvas, cs, doneCallback) {
            var _this = this;
            canvas.animate(200, '>', 0).during(function (t) {
                cs.forEach(function (c, i) {
                    console.log(c);
                    var element = _this.getViewElement(c);
                    if (_this.drawBackend == 1 /* SVG */) {
                        // scale is [0-1], so no need to save startScaleX/startScaleY here
                        var scaleX = SVG.easing.quadOut(t);
                        var scaleY = SVG.easing.quadOut(t);
                        element.scale(scaleX, scaleY);
                    } else if (_this.drawBackend == 0 /* CANVAS */) {
                        // no idea
                    }
                });
            }).after(function () {
                doneCallback();
            });
        };
        return SquareView;
    })();
    View.SquareView = SquareView;
})(View || (View = {}));
/// <reference path="./model.ts" />
/// <reference path="./view.ts" />
var MIN_VAL = 2;

var GameType;
(function (GameType) {
    GameType[GameType["SURVIVAL"] = 0] = "SURVIVAL";
    GameType[GameType["PUZZLE"] = 1] = "PUZZLE";
})(GameType || (GameType = {}));
/// <reference path="./squaregame.ts" />
/// <reference path="./respawnsys.ts" />
var PuzzleRespawnSys = (function () {
    function PuzzleRespawnSys(game) {
        this.game = game;
        this.movesAtLastSpawn = -1;
        this.movesTillNext = this.computeMovesTillNext();
    }
    PuzzleRespawnSys.prototype.computeMovesTillNext = function () {
        return Math.round(20 / this.game.gameParams.level);
    };

    PuzzleRespawnSys.prototype.progressTillNextRespawn = function () {
        return (this.game.gameState.numMoves - this.movesAtLastSpawn) / (this.movesTillNext - this.movesAtLastSpawn);
    };

    PuzzleRespawnSys.prototype.computeNext = function () {
        // first, try to clear imbalances
        var imbalances = this.game.tracker.tiles.map(function (v, i) {
            return v != i && v != 0;
        });
        var tileN = [];
        if (imbalances.some(function (b) {
            return b;
        })) {
            for (var i = MIN_VAL; i < imbalances.length; i++) {
                if (imbalances[i]) {
                    tileN.push(i);
                }
            }
        } else {
            do {
                tileN = [Math.round(Math.random() * (this.game.gameParams.maxVal - MIN_VAL)) + MIN_VAL];
            } while(tileN[0] == this.game.gameState.lastCleared || this.game.grid.size - this.game.gameState.activeCells < tileN[0]);
        }

        return tileN;
    };

    PuzzleRespawnSys.prototype.gameIsStuck = function () {
        return !this.game.tracker.tiles.some(function (n, i) {
            return n >= i;
        });
    };

    PuzzleRespawnSys.prototype.update = function (dt) {
        if (this.game.gameState.numMoves > this.movesTillNext) {
            this.next = this.computeNext(); // this is the only time when this is important
            this.spawnNewTiles();
            this.movesAtLastSpawn = this.movesTillNext;
            this.movesTillNext += this.computeMovesTillNext();
        } else if (this.gameIsStuck()) {
            this.next = this.computeNext(); // this is the only time when this is important
            this.spawnNewTiles();
        }
    };

    PuzzleRespawnSys.prototype.spawnNewTiles = function () {
        var _this = this;
        console.log("spawning new tiles");
        var tileIndices = [];

        // the only sane way to do this is by doing all of the model modifications in a donecallback
        this.next.forEach(function (n) {
            var tilesToPlace = n;

            for (var i = 0; i < _this.game.grid.size; i++) {
                if (_this.game.grid.getFlat(i).value == n) {
                    tilesToPlace--;
                }
            }

            while (tilesToPlace > 0) {
                var insertedPosition;
                if (_this.game.grid.size - _this.game.gameState.activeCells < 10) {
                    var spacesLeft = _this.game.grid.size - _this.game.gameState.activeCells;
                    for (var i = 0; i < _this.game.grid.size; i++) {
                        if (_this.game.grid.getFlat(i).type == 1 /* EMPTY */) {
                            spacesLeft--;
                            if (spacesLeft == 0) {
                                _this.game.grid.setFlat(i, new Tile(2 /* REGULAR */, n));
                                insertedPosition = i;
                            } else if (Math.random() > 0.5) {
                                _this.game.grid.setFlat(i, new Tile(2 /* REGULAR */, n));
                                insertedPosition = i;
                                break;
                            }
                        }
                    }
                } else {
                    do {
                        var randomPlace = Math.round(Math.random() * _this.game.grid.size);
                        while (_this.game.grid.getFlat(randomPlace).type != 1 /* EMPTY */) {
                            randomPlace = Math.round(Math.random() * _this.game.grid.size);
                        }
                        _this.game.grid.setFlat(randomPlace, new Tile(2 /* REGULAR */, n));
                        insertedPosition = randomPlace;
                        var t = _this.game.grid.getFlat(randomPlace);
                        var coords = new CartesianCoords(randomPlace % _this.game.grid.gridw, Math.floor(randomPlace / _this.game.grid.gridw));
                        var canPop = _this.game.grid.floodAcquire(coords).length == t.value;
                    } while(canPop);
                }

                tilesToPlace--;
                _this.game.gameState.activeCells++;
                _this.game.tracker.tiles[n]++;

                var coords = new CartesianCoords(insertedPosition % _this.game.grid.gridw, Math.floor(insertedPosition / _this.game.grid.gridw));
                tileIndices.push(coords);
            }
        });

        // to get rid of flicker, write custom method of syncing view and model that only updates one
        // and set its scale to 0
        this.game.updateViewAndController();

        this.game.gameState.disableHover = true;

        this.game.gridView.materialize(this.game.canvas, tileIndices, function () {
            _this.game.extendUIFromView();
            _this.game.update();
            _this.game.gameState.disableHover = false;
        });
    };
    return PuzzleRespawnSys;
})();
/// <reference path="./squaregame.ts" />
/// <reference path="./respawnsys.ts" />

var SurvivalMechanic = (function () {
    function SurvivalMechanic(game) {
        this.game = game; // need to mutate
        this.params = {
            respawns: 0,
            maxRespawnsTillLimit: 100,
            minRespawnInterval: 3000,
            maxRespawnInterval: 6000
        };
        this.timeSinceLastRespawn = 0;
        this.next = this.computeNext();
    }
    SurvivalMechanic.prototype.timeTillNextRespawn = function () {
        // normalization factor
        var a = (this.params.maxRespawnInterval - this.params.minRespawnInterval) / Math.sqrt(this.params.maxRespawnsTillLimit);
        return a * Math.sqrt(this.params.maxRespawnsTillLimit - this.params.respawns) + this.params.minRespawnInterval;
    };

    SurvivalMechanic.prototype.progressTillNextRespawn = function () {
        return this.timeSinceLastRespawn / this.timeTillNextRespawn();
    };

    SurvivalMechanic.prototype.update = function (dt) {
        if (this.game.gameState.activeCells >= this.game.grid.size - 1)
            return;
        this.timeSinceLastRespawn += dt;
        var interval = this.timeTillNextRespawn();
        if (this.timeSinceLastRespawn >= interval) {
            this.timeSinceLastRespawn = 0;
            this.spawnNewTiles();
            this.params.respawns++;
        }
    };

    SurvivalMechanic.prototype.computeNext = function () {
        var exists = [];
        for (var i = 0; i < this.game.grid.size; i++) {
            exists[this.game.grid.getFlat(i).value] = true;
        }

        var nonexistent = [];
        for (var i = MIN_VAL; i < this.game.gameParams.maxVal; i++) {
            if (exists[i] == null) {
                nonexistent.push(i);
            }
        }

        var tileN = [nonexistent[Math.round(Math.random() * nonexistent.length)]];
        while (tileN[0] == this.game.gameState.lastCleared || this.game.grid.size - this.game.gameState.activeCells < tileN[0]) {
            tileN = [nonexistent[Math.round(Math.random() * nonexistent.length)]];
        }

        return tileN;
    };

    SurvivalMechanic.prototype.spawnNewTiles = function () {
        var _this = this;
        this.next.forEach(function (n) {
            var tilesToPlace = n;
            var tileIndices = [];

            while (tilesToPlace > 0) {
                var insertedPosition;
                if (_this.game.grid.size - _this.game.gameState.activeCells < 10) {
                    var spacesLeft = _this.game.grid.size - _this.game.gameState.activeCells;
                    for (var i = 0; i < _this.game.grid.size; i++) {
                        if (_this.game.grid.getFlat(i).type == 1 /* EMPTY */) {
                            spacesLeft--;
                            if (spacesLeft == 0) {
                                _this.game.grid.setFlat(i, new Tile(2 /* REGULAR */, n));
                                insertedPosition = i;
                            } else if (Math.random() > 0.5) {
                                _this.game.grid.setFlat(i, new Tile(2 /* REGULAR */, n));
                                insertedPosition = i;
                                break;
                            }
                        }
                    }
                } else {
                    var randomPlace = Math.round(Math.random() * _this.game.grid.size);
                    while (_this.game.grid.getFlat(randomPlace).type != 1 /* EMPTY */) {
                        randomPlace = Math.round(Math.random() * _this.game.grid.size);
                    }
                    _this.game.grid.setFlat(randomPlace, new Tile(2 /* REGULAR */, n));
                    insertedPosition = randomPlace;
                }

                tilesToPlace--;
                _this.game.gameState.activeCells++;

                tileIndices.push(insertedPosition);
            }

            _this.game.updateViewAndController();
            _this.game.extendUIFromView();

            tileIndices.forEach(function (i) {
                var c = _this.game.gridView.getViewElements()[i];
                c.scale(0, 0);
                c.animate(200, '>', 0).scale(1, 1).after(function () {
                    _this.game.extendUIFromView();
                });
            });
        });

        this.next = this.computeNext();
    };
    return SurvivalMechanic;
})();
/// <reference path="lib/chroma-js.d.ts" />
/// <reference path="lib/hammerjs.d.ts" />
/// <reference path="lib/svgjs.d.ts" />
/// <reference path="lib/tsm-0.7.d.ts" />
/// <reference path="lib/utils.ts" />
/// <reference path="./game.ts" />
/// <reference path="./puzzlerespawnsys.ts" />
/// <reference path="./survivalmechanic.ts" />
var SquareGameState = (function () {
    function SquareGameState() {
    }
    return SquareGameState;
})();

var SquareGame = (function () {
    function SquareGame(gameType, drawBackend) {
        var _this = this;
        this.timeSinceLastUpdate = new Date().getTime();

        var gp = {
            level: 1,
            gridw: 5,
            gridh: 5,
            maxVal: 5,
            gameType: gameType,
            drawBackend: drawBackend
        };

        if (gp.gameType == 0 /* SURVIVAL */) {
            gp.gridw = 8;
            gp.gridh = 8;
            gp.maxVal = 9;
        }

        this.canvas = View.buildCanvas(640, 650, drawBackend);

        this.init(gp);

        if (drawBackend == 1 /* SVG */) {
            Hammer(this.canvas.node, { preventDefault: true }).on("dragend swipeend", function (e) {
                _this.resolveDrag(e, _this);
            });
        } else if (drawBackend == 0 /* CANVAS */) {
        }
    }
    SquareGame.prototype.updateViewAndController = function () {
        this.gridView.resetView(this.canvas);
        this.extendUIFromView();
    };

    SquareGame.prototype.extendUIOnce = function () {
        var _this = this;
        if (this.gridView.drawBackend == 0 /* CANVAS */) {
            // make sure a mouse:out event is triggered when we move outside of the canvas
            var universe = document.body;
            universe.onmousemove = function (e) {
                return _this.canvas._onMouseMove(e);
            };
            var game = this;
            this.canvas.on("mouse:out", function (options) {
                if (game.gameState.disableHover)
                    return;
                var cell = options.target;

                // if we just mouse out without going over anything, hoverCell remains the same, so clear it
                if (cell.coords.equals(game.gameState.hoverCell))
                    game.gameState.hoverCell = null;
                if (game.grid.get(cell.coords).type == 4 /* DEACTIVATED */ || game.grid.get(cell.coords).type == 1 /* EMPTY */)
                    return;
                var hover = game.grid.floodAcquire(cell.coords);

                // if the current hoverCell is a neighbor of this cell, ignore
                // this is a hack because mouseOut always comes after a mouseOver
                if (hover.some(function (h) {
                    return h.equals(game.gameState.hoverCell);
                }))
                    return;
                hover.forEach(function (t) {
                    var rect = game.gridView.getViewElement(t).rect;
                    rect.fill = game.gridView.colorizer.fromTile(game.grid.get(t));
                    game.canvas.renderAll();
                });
            });
            this.canvas.on("mouse:over", function (options) {
                if (game.gameState.disableHover)
                    return;
                var cell = options.target;
                if (game.grid.get(cell.coords).type == 4 /* DEACTIVATED */ || game.grid.get(cell.coords).type == 1 /* EMPTY */) {
                    game.gameState.hoverCell = null;
                    return;
                }
                game.gameState.hoverCell = cell.coords;
                var hover = game.grid.floodAcquire(cell.coords);
                hover.forEach(function (t) {
                    var rect = game.gridView.getViewElement(t).rect;
                    rect.fill = game.gridView.colorizer.highlightFromTile(game.grid.get(t));
                    _this.canvas.renderAll();
                });
            });
            this.canvas.on("mouse:down", function (options) {
                game.gameState.disableHover = true;
                game.gameState.dragStart = new TSM.vec2([options.e.offsetX, options.e.offsetY]);
                var cell = options.target;
                cell.remove();
                game.canvas.add(cell);
                if (game.grid.get(cell.coords).type != 4 /* DEACTIVATED */) {
                    game.gameState.selected = game.grid.floodAcquire(cell.coords);
                } else {
                    game.gameState.selected = [];
                }
            });
            this.canvas.on("mouse:up", function (options) {
                game.gameState.disableHover = false;
                if (game.gameState.selected == null || game.gameState.selected.length == 0) {
                    console.log("nothing selected");
                    return;
                }

                var upperCanvasNode = document.getElementById("screen-canvas").nextSibling;
                if (options.e.toElement != upperCanvasNode)
                    return;

                var mouse = new TSM.vec2([options.e.offsetX, options.e.offsetY]);
                var delta = TSM.vec2.difference(mouse, game.gameState.dragStart);

                if (Math.abs(delta.x) + Math.abs(delta.y) > Utils.EPSILON) {
                    var moveDirection;

                    if (Math.abs(delta.y) > Math.abs(delta.x)) {
                        moveDirection = delta.y < 0 ? 0 /* NORTH */ : 2 /* SOUTH */;
                    } else {
                        moveDirection = delta.x < 0 ? 3 /* WEST */ : 1 /* EAST */;
                    }

                    game.gameState.selected.forEach(function (t) {
                        game.justMove(t, moveDirection, game);
                    });
                }
            });
        } else {
            // hack fixme
            this.extendUIFromView();
        }
    };

    SquareGame.prototype.extendUIFromView = function () {
        var _this = this;
        if (this.gameParams.drawBackend == 1 /* SVG */) {
            var cells = this.gridView.getViewElements();

            var game = this;
            cells.forEach(function (cell, i) {
                if (cell === null)
                    return;
                cell.mouseover(null).mouseout(null);
                cell.mouseover(function () {
                    if (game.gameState.disableHover)
                        return;
                    if (game.grid.get(cell.coords).type == 4 /* DEACTIVATED */ || game.grid.get(cell.coords).type == 1 /* EMPTY */)
                        return;
                    var hover = game.grid.floodAcquire(cell.coords);
                    hover.forEach(function (t) {
                        game.gridView.getViewElement(t).rect.attr({ 'fill': game.gridView.colorizer.highlightFromTile(game.grid.getFlat(i)) });
                    });
                }).mouseout(function () {
                    if (game.gameState.disableHover)
                        return;
                    var hover = game.grid.floodAcquire(cell.coords);
                    hover.forEach(function (t) {
                        game.gridView.getViewElement(t).rect.attr({ 'fill': game.gridView.colorizer.fromTile(game.grid.getFlat(i)) });
                    });
                });
                var hammer = Hammer(cell.node, { preventDefault: true });

                hammer.on("drag swipe", null);

                hammer.on("drag swipe", function (e) {
                    // 'this' refers to the DOM node directly here
                    if (game.grid.get(cell.coords).type != 4 /* DEACTIVATED */) {
                        game.gameState.selected = game.grid.floodAcquire(cell.coords);
                    } else {
                        game.gameState.selected = [];
                    }

                    // show where we're moving
                    var cellw = game.gridView.getCellSize(game.canvas).w, cellh = game.gridView.getCellSize(game.canvas).h;

                    var moveVector = new CartesianCoords(0, 0);

                    var moved = false;
                    var fudge_epsilon = 10;
                    if (Math.abs(e.gesture.deltaY) > fudge_epsilon || Math.abs(e.gesture.deltaX) > fudge_epsilon) {
                        if (Math.abs(e.gesture.deltaY) > Math.abs(e.gesture.deltaX)) {
                            if (Math.abs(e.gesture.deltaY) > fudge_epsilon) {
                                game.gameState.selected.forEach(function (coord) {
                                    var c = game.gridView.getViewElement(coord);
                                    c.translate(c.cannonicalTransform.x, c.cannonicalTransform.y + e.gesture.deltaY);
                                    c.dragOffset.x = 0;
                                    c.dragOffset.y = e.gesture.deltaY;

                                    var tile = game.grid.get(coord);
                                    if (e.gesture.deltaY > 0) {
                                        var bbs = game.gridView.getViewElement(new CartesianCoords(coord.x, tile.bounds.s));
                                        if (c.cannonicalTransform.y + e.gesture.deltaY > bbs.cannonicalTransform.y) {
                                            c.translate(c.cannonicalTransform.x, bbs.cannonicalTransform.y);
                                            c.dragOffset.y = bbs.cannonicalTransform.y;
                                        }
                                    } else {
                                        var bbn = game.gridView.getViewElement(new CartesianCoords(coord.x, tile.bounds.n));
                                        if (c.cannonicalTransform.y + e.gesture.deltaY < bbn.cannonicalTransform.y) {
                                            c.translate(c.cannonicalTransform.x, bbn.cannonicalTransform.y);
                                            c.dragOffset.y = bbn.cannonicalTransform.y;
                                        }
                                    }
                                });
                            }
                        } else {
                            if (Math.abs(e.gesture.deltaX) > fudge_epsilon) {
                                game.gameState.selected.forEach(function (coord) {
                                    var c = game.gridView.getViewElement(coord);
                                    c.translate(c.cannonicalTransform.x + e.gesture.deltaX, c.cannonicalTransform.y);
                                    c.dragOffset.x = e.gesture.deltaX;
                                    c.dragOffset.y = 0;

                                    var tile = game.grid.get(coord);
                                    if (e.gesture.deltaX > 0) {
                                        var bbe = game.gridView.getViewElement(new CartesianCoords(tile.bounds.e, coord.y));
                                        if (c.cannonicalTransform.x + e.gesture.deltaX > bbe.cannonicalTransform.x) {
                                            c.translate(bbe.cannonicalTransform.x, c.cannonicalTransform.y);
                                            c.dragOffset.x = bbe.cannonicalTransform.x;
                                        }
                                    } else {
                                        var bbw = game.gridView.getViewElement(new CartesianCoords(tile.bounds.w, coord.y));
                                        if (c.cannonicalTransform.x + e.gesture.deltaX < bbw.cannonicalTransform.x) {
                                            c.translate(bbw.cannonicalTransform.x, c.cannonicalTransform.y);
                                            c.dragOffset.x = bbw.cannonicalTransform.x;
                                        }
                                    }
                                });
                            }
                        }
                    } else {
                        game.gameState.selected.forEach(function (coord) {
                            var c = game.gridView.getViewElement(coord);
                            c.translate(c.cannonicalTransform.x, c.cannonicalTransform.y);
                            c.dragOffset.x = 0;
                            c.dragOffset.y = 0;
                        });
                    }
                });
            });
        } else if (this.gameParams.drawBackend == 0 /* CANVAS */) {
            var cells = this.gridView.getViewElements();
            var game = this;
            cells.forEach(function (cell, i) {
                cell.on("moving", function (options) {
                    var mouse = new TSM.vec2([options.e.offsetX, options.e.offsetY]);
                    var delta = TSM.vec2.difference(mouse, _this.gameState.dragStart);
                    var dir = new CartesianCoords(0, 0);
                    var tile = _this.grid.get(cell.coords);
                    var mouseGridCoords = _this.gridView.getGridCoordsFromScreenPos(_this.canvas, mouse);
                    if (Math.abs(delta.x) > Math.abs(delta.y)) {
                        cell.lockMovementY = true;
                        cell.top = cell.cannonicalTop;
                        cell.lockMovementX = false;
                        if (delta.x > 0) {
                            var bbe = _this.gridView.getViewElement(new CartesianCoords(tile.bounds.e, cell.coords.y));

                            // cannonical because bbe can be cell.
                            if (cell.left > bbe.cannonicalLeft) {
                                cell.left = bbe.cannonicalLeft;
                            }
                        } else {
                            var bbw = _this.gridView.getViewElement(new CartesianCoords(tile.bounds.w, cell.coords.y));
                            if (cell.left < bbw.cannonicalLeft) {
                                cell.left = bbw.cannonicalLeft;
                            }
                        }

                        // move group
                        var group = _this.grid.floodAcquire(cell.coords);
                        var dx = cell.left - cell.cannonicalLeft;
                        console.log("dx:" + dx);
                        group.forEach(function (coords) {
                            var tile = _this.gridView.getViewElement(coords);
                            if (!coords.equals(cell.coords))
                                tile.left = tile.cannonicalLeft + dx;
                        });
                    } else {
                        cell.lockMovementX = true;
                        cell.left = cell.cannonicalLeft;
                        cell.lockMovementY = false;
                        if (delta.y > 0) {
                            var bbs = _this.gridView.getViewElement(new CartesianCoords(cell.coords.x, tile.bounds.s));
                            if (cell.top > bbs.cannonicalTop) {
                                cell.top = bbs.cannonicalTop;
                            }
                        } else {
                            var bbn = _this.gridView.getViewElement(new CartesianCoords(cell.coords.x, tile.bounds.n));
                            if (cell.top < bbn.cannonicalTop) {
                                cell.top = bbn.cannonicalTop;
                            }
                        }

                        // move group
                        var group = _this.grid.floodAcquire(cell.coords);
                        var dy = cell.top - cell.cannonicalTop;
                        console.log("dy:" + dy);
                        group.forEach(function (coords) {
                            var tile = _this.gridView.getViewElement(coords);
                            if (!coords.equals(cell.coords))
                                tile.top = tile.cannonicalTop + dy;
                        });
                    }

                    // stupid hack to make sure we don't move out of bounds!!!!
                    var upperCanvasNode = document.getElementById("screen-canvas").nextSibling;
                    if (options.e.toElement != upperCanvasNode) {
                        var offset = Utils.getOffset(upperCanvasNode);
                        if (options.e.x < offset.left) {
                            var bbw = _this.gridView.getViewElement(new CartesianCoords(tile.bounds.w, cell.coords.y));
                            if (cell.left < bbw.cannonicalLeft) {
                                cell.left = bbw.cannonicalLeft;
                            }
                        }
                        if (options.e.y < offset.top) {
                            var bbn = _this.gridView.getViewElement(new CartesianCoords(cell.coords.x, tile.bounds.n));
                            if (cell.top < bbn.cannonicalTop) {
                                cell.top = bbn.cannonicalTop;
                            }
                        }
                    }
                });
            });
        }
    };

    SquareGame.prototype.init = function (gp) {
        var _this = this;
        var gs = {
            activeCells: gp.maxVal * (gp.maxVal + 1) / 2 - 1,
            disableHover: false,
            dragStart: new TSM.vec2([0, 0]),
            hoverEnableCallback: null,
            hoverCell: null,
            selected: [],
            lastCleared: 0,
            numMoves: 0
        };

        this.gameParams = gp;
        this.gameState = gs;

        var tr = {
            tiles: (function () {
                var t = [];
                for (var i = MIN_VAL; i < gp.maxVal; i++) {
                    t[i] = 0;
                }
                return t;
            })()
        };

        this.tracker = tr;

        this.grid = new Model.Square(this.gameParams.gridw, this.gameParams.gridh, new Tile(1 /* EMPTY */, 0), new Tile(0 /* OUT_OF_BOUNDS */, -1));
        this.procGenGrid(this.grid, gp, this.tracker);

        if (gp.drawBackend == 1 /* SVG */)
            this.canvas.clear();

        this.gridView = View.build(this.grid, gp.drawBackend);
        this.updateViewAndController();
        this.extendUIOnce(); // canvas-specific. Rename. Does this require draw?

        switch (this.gameParams.gameType) {
            case 0 /* SURVIVAL */:
                this.survivalMechanic = new SurvivalMechanic(this);
                break;
            case 1 /* PUZZLE */:
                this.respawnSys = new PuzzleRespawnSys(this);
                break;
        }

        if (this.updateID != null) {
            clearInterval(this.updateID);
        }

        this.updateID = setInterval(function () {
            return _this.timedUpdate();
        }, 1000 / 5);
    };

    SquareGame.prototype.timedUpdate = function () {
        var now = new Date().getTime();
        var dt = now - this.timeSinceLastUpdate;
        this.timeSinceLastUpdate = now;

        if (this.gameParams.gameType == 0 /* SURVIVAL */) {
            this.survivalMechanic.update(dt);
            this.gridView.updateTimerBar(this.canvas, this.survivalMechanic.progressTillNextRespawn(), null);
        }

        this.respawnSys.update(dt);
        var changed = this.cachedProgressTillNextRespawn == this.respawnSys.progressTillNextRespawn();
        if (changed) {
            this.gridView.updateTimerBar(this.canvas, this.respawnSys.progressTillNextRespawn(), null);
            this.cachedProgressTillNextRespawn = this.respawnSys.progressTillNextRespawn();
        }
    };

    SquareGame.prototype.update = function (game) {
        if (game == null)
            game = this;

        game.gameState.activeCells = game.updateActiveCells();

        game.grid.recomputeAllBounds();

        if (game.clearedStage()) {
            if (game.gameParams.gameType == 0 /* SURVIVAL */) {
                alert("Holy crap you beat survival mode! How is that even possible. Let's see you do it again.");

                // restart
                game.init(game.gameParams);
            } else {
                game.advance();
            }
        } else if (game.over()) {
            alert("Better luck next time");

            // restart
            game.init(game.gameParams);
        }
    };

    SquareGame.prototype.updateActiveCells = function () {
        var count = 0;
        for (var i = 0; i < this.grid.size; i++) {
            var t = this.grid.getFlat(i);
            if (t.type != 1 /* EMPTY */) {
                count++;
            }
        }
        return count;
    };

    // procedurally generate a playable square grid
    // the grid must:
    // 1) have n of tiles labeled n.
    // 2) have enough empty spaces to be tractable (and fun)
    SquareGame.prototype.procGenGrid = function (grid, gp, tr) {
        if (gp.gameType == 0 /* SURVIVAL */) {
            // decide which tiles we're gonna generate to start this level
            var done = false;
            var acc = 0;
            var toGenerate = [];
            while (!done) {
                var val = Math.round(Math.random() * (gp.maxVal - MIN_VAL) + MIN_VAL);
                while (toGenerate.some(function (v) {
                    return v == val;
                })) {
                    val = Math.round(Math.random() * (gp.maxVal - MIN_VAL) + MIN_VAL);
                }
                if (acc + val < grid.size - 2 * gp.gridw) {
                    toGenerate.push(val);
                    acc += val;
                } else {
                    done = true;
                }
            }

            while (toGenerate.length > 0) {
                var val = toGenerate.pop();
                var added = 0;
                while (added < val) {
                    var randIndex = Math.round(Math.random() * (grid.size - 1));
                    if (grid.getFlat(randIndex).type == 1 /* EMPTY */) {
                        grid.setFlat(randIndex, new Tile(2 /* REGULAR */, val));
                        added++;
                    }
                }
                tr.tiles[val] = val;
            }

            for (var y = 0; y < grid.gridh; y++) {
                for (var x = 0; x < grid.gridw; x++) {
                    var coords = new CartesianCoords(x, y);
                    var t = grid.get(coords);
                    if (this.grid.floodAcquire(coords).length == t.value) {
                        do {
                            grid.set(coords, new Tile(1 /* EMPTY */, -1));
                            var newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            while (grid.get(newCoords).type != 1 /* EMPTY */) {
                                newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            }
                            grid.set(newCoords, t);
                            var canPop = this.grid.floodAcquire(newCoords).length == t.value;
                        } while(canPop);
                    }
                }
            }
        } else {
            var count = [];

            for (var i = 0; i < grid.size; i++) {
                var val = Math.round(Math.random() * (gp.maxVal - MIN_VAL) + MIN_VAL);
                grid.setFlat(i, new Tile(2 /* REGULAR */, val));
                if (count[val] == null) {
                    count[val] = 0;
                }
                count[val]++;
            }

            for (i = 0; i < grid.size; i++) {
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

            for (i = 2; i <= gp.maxVal; i++) {
                if (count[i] < i) {
                    while (count[i] < i) {
                        var randIndex = Math.round(Math.random() * (gp.gridw * gp.gridh - 1));
                        if (grid.getFlat(randIndex).type == 1 /* EMPTY */) {
                            grid.setFlat(randIndex, new Tile(2 /* REGULAR */, i));
                            count[i]++;
                        }
                    }
                }
            }

            for (var y = 0; y < grid.gridh; y++) {
                for (var x = 0; x < grid.gridw; x++) {
                    var coords = new CartesianCoords(x, y);
                    var t = grid.get(coords);
                    if (this.grid.floodAcquire(coords).length == t.value) {
                        do {
                            grid.set(coords, new Tile(1 /* EMPTY */, -1));
                            var newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            while (grid.get(newCoords).type != 1 /* EMPTY */) {
                                newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            }
                            grid.set(newCoords, t);
                            var canPop = this.grid.floodAcquire(newCoords).length == t.value;
                        } while(canPop);
                    }
                }
            }

            for (i = 2; i <= gp.maxVal; i++) {
                if (count[i]) {
                    tr.tiles[i] = count[i];
                } else {
                    console.log("bad state:" + i + " is null");
                }
            }

            this.grid.recomputeAllBounds();
        }

        this.gameState.activeCells = this.updateActiveCells();
    };

    // the only reason why we need game everywhere is because of dumb JavaScript "this" closure definitions!
    // we need from because we need to ignore collisions
    // this should be a function in CartesianCoords, or better yet, use vector!
    SquareGame.prototype.displace = function (set, direction) {
        return set.map(function (cell) {
            return new CartesianCoords(cell.x + direction.x, cell.y + direction.y);
        });
    };

    SquareGame.prototype.justMove = function (tile, direction, game) {
        var _this = this;
        if (game.grid.get(tile).type == 1 /* EMPTY */)
            return;

        var from = game.grid.floodAcquire(tile);
        var to = from.map(function (c) {
            return game.grid.getTileBoundInDirection(c, direction);
        });

        game.gameState.disableHover = true;
        game.gridView.slide(this.canvas, from, to, function () {
             {
                var fromTiles = from.map(function (f) {
                    return game.grid.get(f);
                });
                to.forEach(function (t, i) {
                    if (!to.some(function (h) {
                        return h.equals(from[i]);
                    })) {
                        game.grid.set(from[i], new Tile(1 /* EMPTY */, -1));
                    }
                    game.grid.set(t, new Tile(fromTiles[i].type, fromTiles[i].value));
                });

                // sync
                game.updateViewAndController();
            }

            game.gameState.disableHover = false;
            game.update(game);

             {
                // [ bun ] <- patty <- to
                // crush patty if there is a bun next to it.
                // otherwise, just cascade push the patty
                var patty = to.map(function (c) {
                    return c.displace(direction);
                });
                var bun = patty.map(function (c) {
                    return c.displace(direction);
                });
                game.grid.checkCollision(from, patty).forEach(function (col, i) {
                    if (col && game.grid.get(patty[i]).type != 0 /* OUT_OF_BOUNDS */ && game.grid.get(patty[i]).type != 1 /* EMPTY */) {
                        if (game.grid.get(patty[i]).value < game.grid.get(to[i]).value) {
                            if (game.grid.get(bun[i]).type == 0 /* OUT_OF_BOUNDS */ || game.grid.get(bun[i]).value > game.grid.get(patty[i]).value) {
                                game.gridView.crush(_this.canvas, patty[i], direction, function () {
                                    // after animation, update model
                                    game.tracker.tiles[game.grid.get(patty[i]).value]--;
                                    game.grid.set(patty[i], new Tile(1 /* EMPTY */, -1));
                                    game.updateViewAndController();
                                    game.gameState.lastCleared = game.grid.get(patty[i]).value;
                                    game.update(game);
                                    game.justMove(to[i], direction, game);
                                });
                            } else {
                                game.justMove(patty[i], direction, game);
                            }
                        }
                    }
                });
            }
            game.prune(to[0]);
        });
    };

    SquareGame.prototype.resolveDrag = function (e, game) {
        if (game.gameState.selected == null || game.gameState.selected.length == 0) {
            console.log("nothing selected");
            return;
        }

        if (game.gameState.disableHover)
            return;

        var moveDirection;

        var fudge_epsilon = 10;
        if (Math.abs(e.gesture.deltaY) > fudge_epsilon || Math.abs(e.gesture.deltaX) > fudge_epsilon) {
            if (Math.abs(e.gesture.deltaY) > Math.abs(e.gesture.deltaX)) {
                moveDirection = e.gesture.deltaY < 0 ? 0 /* NORTH */ : 2 /* SOUTH */;
            } else {
                moveDirection = e.gesture.deltaX < 0 ? 3 /* WEST */ : 1 /* EAST */;
            }
            game.justMove(game.gameState.selected[0], moveDirection, game);
        }
    };

    SquareGame.prototype.over = function () {
        return this.gameState.activeCells >= this.grid.size - 1;
    };

    SquareGame.prototype.clearedStage = function () {
        console.log("checking win condition: " + this.gameState.activeCells + " cells left.");
        return this.gameState.activeCells == 0;
    };

    SquareGame.prototype.advance = function () {
        var gp = Utils.deepCopy(this.gameParams);
        gp.level++;

        gp.gridw = [5, 8, 10].reduce(function (prev, cur, i, array) {
            if (prev > gp.gridw)
                return prev;
            else if (cur > gp.gridw)
                return cur;
            else if (i == array.length - 1)
                return cur;
        });
        gp.gridh = gp.gridw;

        gp.maxVal = Math.floor(Math.sqrt(gp.gridw * gp.gridh));
        if (gp.maxVal > 9)
            gp.maxVal = 9;

        this.gridView.updateTimerBar(null, null, null);
        this.init(gp);
    };

    SquareGame.prototype.prune = function (start) {
        var _this = this;
        // see if we should delete this cell and surrounding cells
        var startTile = this.grid.get(start);
        if (startTile.type == 4 /* DEACTIVATED */ || startTile.type == 1 /* EMPTY */)
            return;
        var targets = this.grid.floodAcquire(start);
        if (targets.length >= startTile.value) {
            console.log("pruning " + targets.length + " tiles");
            var str = "";
            targets.forEach(function (t) {
                str += _this.grid.get(t).value + " ";
            });
            console.log(str);
            var game = this;
            this.gridView.pop(this.canvas, targets, function () {
                // after animation, update model
                targets.forEach(function (t, i) {
                    game.tracker.tiles[startTile.value]--;
                    game.grid.set(t, new Tile(1 /* EMPTY */, -1));
                });
                game.gameState.lastCleared = startTile.value;
                game.updateViewAndController();
                game.update(game);
            });
        }
    };

    SquareGame.prototype.debugPrintView = function () {
        return this.gridView.debugPrint();
    };

    SquareGame.prototype.debugPrintModel = function () {
        return this.grid.debugPrint();
    };
    return SquareGame;
})();
/// <reference path="./squaregame.ts" />
/// <reference path="lib/utils.ts" />
var game;

var branch = function () {
    if (document.cookie.match(/^(.*;)?tess_prototype_seen_splash=[^;]+(.*)?$/)) {
        location.href = "play.html";
    }
};

var gotoGame = function () {
    var expires = new Date();
    expires.setFullYear(expires.getFullYear() + 1);
    var cookie = "tess_prototype_seen_splash=true; path=/tessellations/; expires=" + expires.toUTCString();
    document.cookie = cookie;
    location.href = "play.html";
};

var init = function () {
    var doSurvival = Utils.getURLParameter("survival") == "true";
    var gameMode = doSurvival ? 0 /* SURVIVAL */ : 1 /* PUZZLE */;
    game = new SquareGame(gameMode, 1 /* SVG */);
};
