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


var AxialCoords = (function () {
    function AxialCoords(q, r) {
        this.q = q;
        this.r = r;
    }
    return AxialCoords;
})();

var CartesianCoords = (function () {
    function CartesianCoords(x, y) {
        this.x = x;
        this.y = y;
    }
    return CartesianCoords;
})();

var Tile = (function () {
    function Tile(t, v) {
        this.type = t;
        this.value = v;
        this.selected = false;
    }
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
        return Square;
    })();
    Model.Square = Square;

    var Hex = (function () {
        function Hex(gridr, DefaultTile, OutOfBoundsTile) {
            this.grid = [];
            this.gridr = gridr;
            this.size = this.diameter() * this.diameter();
            this.outOfBoundsTile = OutOfBoundsTile;
            for (var r = -this.gridr; r <= this.gridr; r++) {
                for (var q = -this.gridr; q <= this.gridr; q++) {
                    if (Math.abs(r + q) > this.gridr)
                        this.grid.push(OutOfBoundsTile);
                    else
                        this.grid.push(DefaultTile);
                }
            }
        }
        // we need to make sure diameter is always odd...
        // in hex grid units
        Hex.prototype.diameter = function () {
            return (this.gridr * 2 + 1);
        };

        // q, r are relative to the center (IE: ( gridr, gridr ) in grid), convert it into the absolute index
        Hex.prototype.toFlat = function (q, r) {
            var x = r + this.gridr;
            var z = q + this.gridr;
            return x + z * (this.diameter());
        };

        Hex.prototype.get = function (c) {
            return Math.abs(c.q) <= this.gridr && Math.abs(c.r) <= this.gridr ? this.grid[this.toFlat(c.q, c.r)] : this.outOfBoundsTile;
        };

        Hex.prototype.getFlat = function (i) {
            return i >= 0 && i < this.grid.length ? this.grid[i] : this.outOfBoundsTile;
        };

        Hex.prototype.set = function (c, tile) {
            if (Math.abs(c.q) <= this.gridr && Math.abs(c.r) <= this.gridr) {
                this.grid[this.toFlat(c.q, c.r)] = tile;
            }
        };

        Hex.prototype.setFlat = function (i, tile) {
            if (i >= 0 && i < this.grid.length) {
                this.grid[i] = tile;
            }
        };

        Hex.prototype.getTileArray = function () {
            return this.grid;
        };
        return Hex;
    })();
    Model.Hex = Hex;
})(Model || (Model = {}));
/// <reference path="lib/chroma-js.d.ts" />
/// <reference path="lib/svgjs.d.ts" />
/// <reference path="./model.ts" />

var Colorizer = (function () {
    function Colorizer() {
        this.highlightFromTile = function (t) {
            if (t.type == 1 /* EMPTY */) {
                return this.scale[t.type](t.value / 9).hex();
            } else {
                return this.scale[t.type](t.value / 9).brighter().hex();
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
    function fromModel(grid) {
        if (grid instanceof Model.Square)
            return new SquareView(grid);
        else if (grid instanceof Model.Hex)
            return new HexView(grid);
    }
    View.fromModel = fromModel;

    var SquareView = (function () {
        function SquareView(square) {
            this.model = square;
            this.colorizer = new Colorizer();
        }
        SquareView.prototype.drawTile = function (canvas, x, y, e) {
            var cellw = canvas.width() / this.model.gridw, cellh = canvas.width() / this.model.gridh;

            var xOffset = cellw / 2;
            var yOffset = cellh / 2;

            var cell = canvas.group().transform({ x: x * cellw + xOffset, y: y * cellh + yOffset });

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
            cell.cannonicalTransform = { x: cell.transform('x'), y: cell.transform('y') };

            return cell;
        };

        SquareView.prototype.updateTimerBar = function (canvas, frac, color) {
            if (canvas == null) {
                this.timerBar.remove();
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
        };

        SquareView.prototype.resetView = function (canvas, forceReset) {
            if (this.model == null)
                return;
            if (this.cells == null)
                this.cells = [];

            for (var y = 0; y < this.model.gridh; y++) {
                for (var x = 0; x < this.model.gridw; x++) {
                    var e = this.model.get(new CartesianCoords(x, y));
                    if (forceReset || this.cells[this.model.toFlat(x, y)] == null || this.cells[this.model.toFlat(x, y)].e != e) {
                        if (forceReset || this.cells[this.model.toFlat(x, y)] != null) {
                            this.cells[this.model.toFlat(x, y)].clear();
                        }
                        this.cells[this.model.toFlat(x, y)] = this.drawTile(canvas, x, y, e);
                    }
                }
            }
        };

        SquareView.prototype.getSVGElements = function () {
            return this.cells;
        };

        SquareView.prototype.getSVGElement = function (c) {
            return c.y >= 0 && c.x >= 0 && c.x < this.model.gridw && c.y < this.model.gridh ? this.cells[this.model.toFlat(c.x, c.y)] : null;
        };
        return SquareView;
    })();
    View.SquareView = SquareView;

    var HexView = (function () {
        function HexView(hex) {
            this.model = hex;
            this.colorizer = new Colorizer();
        }
        HexView.prototype.toAbsX = function (q) {
            return q + this.model.gridr;
        };

        HexView.prototype.toAbsY = function (r) {
            return r + this.model.gridr;
        };

        HexView.prototype.drawTile = function (canvas, q, r, e) {
            var fudgeMargin = 10;
            var size = (canvas.attr("width") - fudgeMargin) / this.model.diameter();

            // size is the diameter of the inscribed circle. radius refers to the radius of the
            // circumscribed circle
            var radius = size / Math.sin(60 * Math.PI / 180) / 2;

            // distance between adjacent (heightwise) hex cells
            var yd = radius * Math.cos(60 * Math.PI / 180);

            // distance between adjacent (withwise) hex cells
            var xd = radius * Math.sin(60 * Math.PI / 180);

            var yOffset = radius;
            var xOffset = xd;

            var x = this.toAbsX(q);
            var y = this.toAbsY(r);

            var center_x = size * x + xd * r + xOffset;
            var center_y = (yd + radius) * y + yOffset;

            var cell = canvas.group().transform({ x: center_x, y: center_y });

            var pts = [new Vec2(0, -radius), new Vec2(xd, -yd), new Vec2(xd, yd), new Vec2(0, radius), new Vec2(-xd, yd), new Vec2(-xd, -yd)];
            var ptstr = pts.reduce(function (p1, p2, i, v) {
                return p1.toString() + " " + p2.toString();
            }, "");

            var hex = cell.polygon(ptstr);

            hex.attr({
                'fill': this.colorizer.fromTile(e),
                'stroke': '#fff',
                'stroke-width': 2 });

            var text = cell.plain(e.type != 1 /* EMPTY */ ? e.value.toString() : "");
            text.attr({
                'fill': this.colorizer.foregroundFromColor(this.colorizer.fromTile(e)),
                'font-size': radius / 1.5 }).transform({ x: -text.attr('font-size') / 3, y: text.attr('font-size') / 3 }); // offset starting point, which is bottom left

            // cache hooks for UI
            cell.coords = new AxialCoords(q, r);
            cell.hex = hex;

            return cell;
        };

        HexView.prototype.updateTimerBar = function (canvas, percentage, color) {
        };

        HexView.prototype.resetView = function (canvas, forceReset) {
            if (this.model == null)
                return;

            this.cells = [];

            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(new AxialCoords(q, r));
                    if (e.type == 0 /* OUT_OF_BOUNDS */) {
                        this.cells[this.model.toFlat(q, r)] = null; // standin
                    }
                }
            }

            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(new AxialCoords(q, r));
                    if (e.type == 1 /* EMPTY */) {
                        this.cells[this.model.toFlat(q, r)] = this.drawTile(canvas, q, r, e);
                    } else if (e.type != 0 /* OUT_OF_BOUNDS */) {
                        // just draw an empty tile anyway
                        this.drawTile(canvas, q, r, new Tile(1 /* EMPTY */, -1));
                    }
                }
            }

            for (var r = -this.model.gridr; r <= this.model.gridr; r++) {
                for (var q = -this.model.gridr; q <= this.model.gridr; q++) {
                    var e = this.model.get(new AxialCoords(q, r));
                    if (e.type == 2 /* REGULAR */) {
                        this.cells[this.model.toFlat(q, r)] = this.drawTile(canvas, q, r, e);
                    }
                }
            }
        };

        HexView.prototype.getSVGElements = function () {
            return this.cells;
        };

        HexView.prototype.getSVGElement = function (c) {
            return Math.abs(c.q) <= this.model.gridr && Math.abs(c.r) <= this.model.gridr ? this.cells[this.model.toFlat(c.q, c.r)] : null;
        };
        return HexView;
    })();
    View.HexView = HexView;
})(View || (View = {}));
/// <reference path="./model.ts" />
/// <reference path="./view.ts" />
var MIN_VAL = 2;

var GameType;
(function (GameType) {
    GameType[GameType["SURVIVAL"] = 0] = "SURVIVAL";
    GameType[GameType["PUZZLE"] = 1] = "PUZZLE";
})(GameType || (GameType = {}));
var Utils = (function () {
    function Utils() {
    }
    Utils.deepCopy = function (src) {
        var dst = {};
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
    return Utils;
})();
/// <reference path="./game.ts" />
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

            _this.game.draw();
            _this.game.extendUI();

            tileIndices.forEach(function (i) {
                var c = _this.game.gridView.getSVGElements()[i];
                c.scale(0, 0);
                c.animate(200, '>', 0).scale(1, 1).after(function () {
                    _this.game.extendUI();
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
/// <reference path="lib/utils.ts" />
/// <reference path="./game.ts" />
/// <reference path="./survivalmechanic.ts" />
var HexGame = (function () {
    function HexGame(gameType, canvas) {
        var _this = this;
        this.timeSinceLastUpdate = new Date().getTime();
        if (canvas == null) {
            canvas = SVG('screen').size(720, 720);
            console.log("no canvas supplied, starting with default canvas with dims " + canvas.width() + ", " + canvas.height());
        }

        this.canvas = canvas;

        var gp = {
            level: 1,
            gridw: 6,
            gridh: 6,
            maxVal: 5,
            gameType: gameType
        };

        if (gp.gameType == 0 /* SURVIVAL */) {
            gp.gridw = 10;
            gp.gridh = 10;
            gp.maxVal = 9;
            this.survivalMechanic = new SurvivalMechanic(this);
        }

        this.init(gp);

        Hammer(this.canvas.node, { preventDefault: true }).on("dragend swipeend", function (e) {
            _this.onDrag(e, _this);
        });

        setInterval(function () {
            return _this.update(_this);
        }, 1000 / 5);
    }
    HexGame.prototype.draw = function () {
        this.canvas.clear();
        this.gridView.resetView(this.canvas);
    };

    HexGame.prototype.extendUI = function () {
        var cells = this.gridView.getSVGElements();

        var game = this;
        cells.forEach(function (cell, i) {
            if (cell === null)
                return;
            cell.mouseover(function () {
                if (game.gameState.disableMouse)
                    return;

                // 'game' refers to a wrapper provided by SVGjs, so we have to go down to node to get the model
                if (game.grid.get(cell.coords).type == 4 /* DEACTIVATED */)
                    return;
                var hover = game.floodAcquire(cell.coords, game.grid.get(cell.coords));
                hover.forEach(function (t) {
                    game.gridView.getSVGElement(t).hex.attr({ fill: game.gridView.colorizer.highlightFromTile(game.grid.getFlat(i)) });
                });
            }).mouseout(function () {
                if (game.gameState.disableMouse)
                    return;
                var hover = game.floodAcquire(cell.coords, game.grid.get(cell.coords));
                hover.forEach(function (t) {
                    game.gridView.getSVGElement(t).hex.attr({ fill: game.gridView.colorizer.fromTile(game.grid.getFlat(i)) });
                });
            });
            var hammer = Hammer(cell.node, { preventDefault: true });
            hammer.on("dragstart swipestart", function (e) {
                if (game.grid.get(cell.coords).type != 4 /* DEACTIVATED */) {
                    game.gameState.selected = game.floodAcquire(cell.coords, game.grid.get(cell.coords));
                } else {
                    game.gameState.selected = [];
                }
            });
        });
    };

    HexGame.prototype.update = function (game) {
        if (game == null)
            game = this;

        var now = new Date().getTime();
        var dt = now - this.timeSinceLastUpdate;
        game.timeSinceLastUpdate = now;

        if (game.clearedStage()) {
            if (game.gameParams.gameType == 0 /* SURVIVAL */) {
                alert("Holy crap you beat survival mode! How is that even possible");
            } else {
                game.advance();
            }
        }

        if (game.gameParams.gameType == 0 /* SURVIVAL */) {
            game.survivalMechanic.update(dt);
        }
    };

    HexGame.prototype.init = function (gp) {
        var gs = {
            activeCells: gp.maxVal * (gp.maxVal + 1) / 2 - 1,
            disableMouse: false,
            selected: [],
            lastCleared: 0,
            numMoves: 0
        };

        this.grid = new Model.Hex(Math.floor(gp.gridw / 2), new Tile(1 /* EMPTY */, 0), new Tile(0 /* OUT_OF_BOUNDS */, -1));

        this.gameParams = gp;
        this.gameState = gs;

        this.procGenGrid(this.grid, gp);
        this.gridView = View.fromModel(this.grid);

        this.draw();
        this.extendUI();
    };

    // procedurally generate a playable hex grid
    // the grid must:
    // 1) have n of tiles labeled n.
    // 2) have enough empty spaces to be tractable (and fun)
    HexGame.prototype.procGenGrid = function (grid, gp) {
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
            if (count[i] < i || count[i] % i != 0) {
                while (count[i] < i || count[i] % i != 0) {
                    var randIndex = Math.round(Math.random() * (grid.getTileArray().length - 1));
                    if (grid.getFlat(randIndex).type == 1 /* EMPTY */) {
                        grid.setFlat(randIndex, new Tile(2 /* REGULAR */, i));
                        count[i]++;
                    }
                }
            }
        }
    };

    HexGame.prototype.floodAcquire = function (start, tile) {
        var cluster = [];
        var marked = { get: null, set: null };

        marked.get = function (key) {
            return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
        };
        marked.set = function (key) {
            this[JSON.stringify(key)] = true;
        };

        var Q = [];
        if (this.grid.get(new AxialCoords(start.q, start.r)) != tile)
            return [];
        Q.push(start);
        while (Q.length > 0) {
            var n = Q.shift();
            if (this.grid.get(n).value == tile.value && this.grid.get(n).type == tile.type && !marked.get(n)) {
                var w = new AxialCoords(n.q, n.r);
                var e = new AxialCoords(n.q, n.r);

                while (this.grid.get(new AxialCoords(w.q - 1, w.r)).value == tile.value && this.grid.get(new AxialCoords(w.q - 1, w.r)).type == tile.type) {
                    w.q--;
                }

                while (this.grid.get(new AxialCoords(e.q + 1, e.r)).value == tile.value && this.grid.get(new AxialCoords(e.q + 1, e.r)).type == tile.type) {
                    e.q++;
                }

                for (var q = w.q; q <= e.q; q++) {
                    var nn = new AxialCoords(q, n.r);
                    marked.set(nn);
                    cluster.push(nn);

                    var nw = new AxialCoords(nn.q, nn.r - 1);
                    var ne = new AxialCoords(nn.q + 1, nn.r - 1);
                    var sw = new AxialCoords(nn.q - 1, nn.r + 1);
                    var se = new AxialCoords(nn.q, nn.r + 1);

                    if (this.grid.get(nw).value == tile.value && this.grid.get(nw).type == tile.type)
                        Q.push(nw);
                    if (this.grid.get(ne).value == tile.value && this.grid.get(ne).type == tile.type)
                        Q.push(ne);
                    if (this.grid.get(sw).value == tile.value && this.grid.get(sw).type == tile.type)
                        Q.push(sw);
                    if (this.grid.get(se).value == tile.value && this.grid.get(se).type == tile.type)
                        Q.push(se);
                }
            }
        }
        return cluster;
    };

    HexGame.prototype.prune = function (start, postCallback) {
        this.draw(); // sync model and view/DOM elements -- this is important, otherwise the view will be outdated and animations won't play right!!!
        var startTile = this.grid.get(start);
        var group = this.floodAcquire(start, startTile);
        if (group.length >= startTile.value) {
            if (startTile.type == 4 /* DEACTIVATED */)
                return;
            var game = this;
            group.forEach(function (cell, i) {
                if (i >= startTile.value)
                    return;

                // each Tile should have its own "prune" behavior
                if (game.grid.get(cell).type == 2 /* REGULAR */) {
                    game.grid.set(cell, new Tile(1 /* EMPTY */, -1));
                    var c = game.gridView.getSVGElement(cell);
                    c.animate(200, '>', 0).scale(0, 0).after(function () {
                        game.gameState.activeCells--;
                        if (i == startTile.value - 1) {
                            postCallback();
                            game.gameState.lastCleared = startTile.value;
                        }
                    });
                } else if (game.grid.get(cell).type == 5 /* LAVA */) {
                    game.grid.set(cell, new Tile(4 /* DEACTIVATED */, game.grid.get(cell).value));
                }
            });
        } else {
            postCallback();
        }
    };

    HexGame.prototype.clearedStage = function () {
        return this.gameState.activeCells == 0;
    };

    HexGame.prototype.advance = function () {
        var gp = Utils.deepCopy(this.gameParams);
        gp.level++;

        gp.gridw = Math.floor((gp.level + 4) * 1.2);
        gp.gridh = Math.floor((gp.level + 4) * 1.2);

        gp.maxVal = gp.level + 3;

        this.init(gp);
    };

    HexGame.prototype.onDrag = function (e, game) {
        var selected = game.gameState.selected;
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
                return new AxialCoords(cell.q + direction.q, cell.r + direction.r);
            });
        }

        function checkCollision(newset, oldset) {
            return newset.map(function (cell, i) {
                // if cell is out of bounds, then, collision
                // if cell is not in original set and cell is not -1 then collision
                // if cell is not in original set and cell is -1 then no collision
                // if cell is in original set then no collsion
                var cellIsOutofBounds = game.grid.get(cell).type == 0 /* OUT_OF_BOUNDS */;
                var cellInOldSet = oldset.some(function (c) {
                    return c.q == cell.q && c.r == cell.r;
                });
                var isCollision = cellIsOutofBounds || (!cellInOldSet && game.grid.get(cell).type != 1 /* EMPTY */);
                return isCollision;
            });
        }

        function move(from, to) {
            // cache all the from values before clearing them
            var fromVals = from.map(function (cell) {
                return game.grid.get(cell);
            });
            from.forEach(function (cell) {
                game.grid.set(cell, new Tile(1 /* EMPTY */, -1));
            });
            to.forEach(function (cell, i) {
                game.grid.set(cell, new Tile(fromVals[i].type, fromVals[i].value));
            });

            for (var i = 0; i < from.length; i++) {
                var f = game.gridView.getSVGElement(from[i]);
                var t = game.gridView.getSVGElement(to[i]);

                game.gameState.disableMouse = true;

                var anim = f.animate(100, '>', 0).move(t.transform('x'), t.transform('y'));

                if (i == 0) {
                    anim.after(function () {
                        game.gameState.disableMouse = false;
                        game.prune(to[0], function () {
                            game.draw();
                            game.extendUI();
                            game.update();
                        });
                    });
                }
            }
        }

        if (game.grid.get(selected[0]).type == 1 /* EMPTY */) {
            return;
        }

        game.prune(selected[0], function () {
            game.draw();
            game.extendUI();
            game.update();
        });

        if (game.grid.get(selected[0]).type == 1 /* EMPTY */) {
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

        game.gameState.selected = selected;
    };
    return HexGame;
})();
/// <reference path="./game.ts" />
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

    PuzzleRespawnSys.prototype.update = function (dt) {
        if (this.game.gameState.numMoves > this.movesTillNext) {
            this.next = this.computeNext(); // this is the only time when this is important
            this.spawnNewTiles();
            this.movesAtLastSpawn = this.movesTillNext;
            this.movesTillNext += this.computeMovesTillNext();
        }
    };

    PuzzleRespawnSys.prototype.floodAcquire = function (start, tile) {
        var cluster = [];
        var marked = { get: null, set: null };
        marked.get = function (key) {
            return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
        };
        marked.set = function (key) {
            this[JSON.stringify(key)] = true;
        };
        var Q = [];
        if (this.game.grid.get(new CartesianCoords(start.x, start.y)) != tile)
            return [];
        Q.push(start);
        while (Q.length > 0) {
            var n = Q.shift();
            if (this.game.grid.get(n).value == tile.value && this.game.grid.get(n).type == tile.type && !marked.get(n)) {
                var w = new CartesianCoords(n.x, n.y);
                var e = new CartesianCoords(n.x, n.y);

                while (this.game.grid.get(new CartesianCoords(w.x - 1, w.y)).value == tile.value && this.game.grid.get(new CartesianCoords(w.x - 1, w.y)).type == tile.type) {
                    w.x--;
                }

                while (this.game.grid.get(new CartesianCoords(e.x + 1, e.y)).value == tile.value && this.game.grid.get(new CartesianCoords(e.x + 1, e.y)).type == tile.type) {
                    e.x++;
                }

                for (var x = w.x; x < e.x + 1; x++) {
                    var nn = new CartesianCoords(x, n.y);
                    marked.set(nn);
                    cluster.push(nn);
                    var north = new CartesianCoords(nn.x, nn.y - 1);
                    var south = new CartesianCoords(nn.x, nn.y + 1);
                    if (this.game.grid.get(north).value == tile.value && this.game.grid.get(north).type == tile.type)
                        Q.push(north);
                    if (this.game.grid.get(south).value == tile.value && this.game.grid.get(south).type == tile.type)
                        Q.push(south);
                }
            }
        }
        return cluster;
    };

    PuzzleRespawnSys.prototype.spawnNewTiles = function () {
        var _this = this;
        var tileIndices = [];
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
                        var canPop = _this.floodAcquire(new CartesianCoords(_this.game.grid.size % randomPlace, Math.floor(_this.game.grid.size / randomPlace)), t).length == t.value;
                    } while(canPop);
                }

                tilesToPlace--;
                _this.game.gameState.activeCells++;
                _this.game.tracker.tiles[n]++;

                tileIndices.push(insertedPosition);
            }
        });

        this.game.draw();
        this.game.extendUI();

        tileIndices.forEach(function (i) {
            var c = _this.game.gridView.getSVGElements()[i];
            c.scale(0, 0);
            c.animate(200, '>', 0).scale(1, 1).after(function () {
                _this.game.extendUI();
            });
        });
    };
    return PuzzleRespawnSys;
})();
/// <reference path="lib/chroma-js.d.ts" />
/// <reference path="lib/hammerjs.d.ts" />
/// <reference path="lib/svgjs.d.ts" />
/// <reference path="lib/tsm-0.7.d.ts" />
/// <reference path="lib/utils.ts" />
/// <reference path="./game.ts" />
/// <reference path="./puzzlerespawnsys.ts" />
/// <reference path="./survivalmechanic.ts" />
var CardinalDirection;
(function (CardinalDirection) {
    CardinalDirection[CardinalDirection["NORTH"] = 0] = "NORTH";
    CardinalDirection[CardinalDirection["EAST"] = 1] = "EAST";
    CardinalDirection[CardinalDirection["SOUTH"] = 2] = "SOUTH";
    CardinalDirection[CardinalDirection["WEST"] = 3] = "WEST";
})(CardinalDirection || (CardinalDirection = {}));

var SquareGame = (function () {
    function SquareGame(gameType, canvas) {
        var _this = this;
        this.timeSinceLastUpdate = new Date().getTime();
        if (canvas == null) {
            canvas = SVG('screen').size(640, 650);
            console.log("no canvas supplied, starting with default canvas with dims " + canvas.width() + ", " + canvas.height());
        }

        this.canvas = canvas;

        var gp = {
            level: 1,
            gridw: 5,
            gridh: 5,
            maxVal: 6,
            gameType: gameType
        };

        if (gp.gameType == 0 /* SURVIVAL */) {
            gp.gridw = 8;
            gp.gridh = 8;
            gp.maxVal = 9;
        }

        this.init(gp);

        Hammer(this.canvas.node, { preventDefault: true }).on("dragend swipeend", function (e) {
            _this.onDrag(e, _this);
        });
    }
    SquareGame.prototype.draw = function () {
        this.gridView.resetView(this.canvas);
    };

    SquareGame.prototype.extendUI = function () {
        var cells = this.gridView.getSVGElements();

        var game = this;
        cells.forEach(function (cell, i) {
            if (cell === null)
                return;
            cell.mouseover(null).mouseout(null);
            cell.mouseover(function () {
                if (game.gameState.disableMouse)
                    return;
                if (game.grid.get(cell.coords).type == 4 /* DEACTIVATED */ || game.grid.get(cell.coords).type == 1 /* EMPTY */)
                    return;
                var hover = game.floodAcquire(cell.coords, game.grid.get(cell.coords));
                hover.forEach(function (t) {
                    game.gridView.getSVGElement(t).rect.attr({ 'fill': game.gridView.colorizer.highlightFromTile(game.grid.getFlat(i)) });
                });
            }).mouseout(function () {
                if (game.gameState.disableMouse)
                    return;
                var hover = game.floodAcquire(cell.coords, game.grid.get(cell.coords));
                hover.forEach(function (t) {
                    game.gridView.getSVGElement(t).rect.attr({ 'fill': game.gridView.colorizer.fromTile(game.grid.getFlat(i)) });
                });
            });
            function displace(set, direction) {
                return set.map(function (cell) {
                    return new CartesianCoords(cell.x + direction.x, cell.y + direction.y);
                });
            }
            function checkCollision(newset, oldset) {
                return newset.map(function (cell, i) {
                    // if cell is out of bounds, then, collision
                    // if cell is not in original set and cell is not -1 then collision
                    // if cell is not in original set and cell is -1 then no collision
                    // if cell is in original set then no collsion
                    var cellIsOutofBounds = game.grid.get(cell).type == 0 /* OUT_OF_BOUNDS */;
                    var cellInOldSet = oldset.some(function (c) {
                        return c.x == cell.x && c.y == cell.y;
                    });
                    var isCollision = cellIsOutofBounds || (!cellInOldSet && game.grid.get(cell).type != 1 /* EMPTY */);
                    return isCollision;
                });
            }
            var hammer = Hammer(cell.node, { preventDefault: true });

            hammer.on("drag swipe", null);

            hammer.on("drag swipe", function (e) {
                // 'this' refers to the DOM node directly here
                if (game.grid.get(cell.coords).type != 4 /* DEACTIVATED */) {
                    game.gameState.selected = game.floodAcquire(cell.coords, game.grid.get(cell.coords));
                } else {
                    game.gameState.selected = [];
                }

                // show where we're moving
                var fudge_epsilon = 20;
                var cellw = Math.floor(game.canvas.width() / game.gameParams.gridw), cellh = Math.floor(game.canvas.width() / game.gameParams.gridh);

                var dragOffsetX = cellw / 3;
                var dragOffsetY = cellh / 3;
                var dragOffset = { x: 0, y: 0 };
                var moveVector = { x: 0, y: 0 };

                var moved = false;
                if (Math.abs(e.gesture.deltaY) > fudge_epsilon || Math.abs(e.gesture.deltaX) > fudge_epsilon) {
                    if (Math.abs(e.gesture.deltaY) > Math.abs(e.gesture.deltaX)) {
                        if (Math.abs(e.gesture.deltaY) > fudge_epsilon) {
                            dragOffset.y = e.gesture.deltaY < 0 ? -dragOffsetY : dragOffsetY;
                            moveVector.y = e.gesture.deltaY < 0 ? -1 : 1;
                            moved = true;
                        }
                    } else {
                        if (Math.abs(e.gesture.deltaX) > fudge_epsilon) {
                            dragOffset.x = e.gesture.deltaX < 0 ? -dragOffsetX : dragOffsetX;
                            moveVector.x = e.gesture.deltaX < 0 ? -1 : 1;
                            moved = true;
                        }
                    }
                }

                if (moved) {
                    var future = displace(game.gameState.selected, moveVector);
                    if (checkCollision(future, game.gameState.selected).every(function (col) {
                        return col == false;
                    })) {
                        game.gameState.selected.forEach(function (coord) {
                            var c = game.gridView.getSVGElement(coord);
                            c.animate(100, '>', 0).move(c.cannonicalTransform.x + dragOffset.x, c.cannonicalTransform.y + dragOffset.y);
                        });
                    }
                } else {
                    game.gameState.selected.forEach(function (coord) {
                        var c = game.gridView.getSVGElement(coord);
                        c.animate(100, '>', 0).move(c.cannonicalTransform.x, c.cannonicalTransform.y);
                    });
                }
            });
        });
    };

    SquareGame.prototype.init = function (gp) {
        var _this = this;
        var gs = {
            activeCells: gp.maxVal * (gp.maxVal + 1) / 2 - 1,
            disableMouse: false,
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

        this.canvas.clear();
        this.gridView = View.fromModel(this.grid);
        this.draw();
        this.extendUI();

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
            return _this.update(_this);
        }, 1000 / 5);
    };

    SquareGame.prototype.update = function (game) {
        if (game == null)
            game = this;

        var now = new Date().getTime();
        var dt = now - game.timeSinceLastUpdate;
        game.timeSinceLastUpdate = now;

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

        if (game.gameParams.gameType == 0 /* SURVIVAL */) {
            game.survivalMechanic.update(dt);
            game.gridView.updateTimerBar(game.canvas, game.survivalMechanic.progressTillNextRespawn(), null);
        } else {
            game.respawnSys.update(dt);
            game.gridView.updateTimerBar(game.canvas, game.respawnSys.progressTillNextRespawn(), null);
        }
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

            // add necessary elements
            this.gameState.activeCells = 0;
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
                this.gameState.activeCells += val;
            }

            for (var y = 0; y < grid.gridh; y++) {
                for (var x = 0; x < grid.gridw; x++) {
                    var coords = new CartesianCoords(x, y);
                    var t = grid.get(coords);
                    if (this.floodAcquire(coords, t).length == t.value) {
                        do {
                            grid.set(coords, new Tile(1 /* EMPTY */, -1));
                            var newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            while (grid.get(newCoords).type != 1 /* EMPTY */) {
                                newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            }
                            grid.set(newCoords, t);
                            var canPop = this.floodAcquire(newCoords, t).length == t.value;
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
                    if (this.floodAcquire(coords, t).length == t.value) {
                        do {
                            grid.set(coords, new Tile(1 /* EMPTY */, -1));
                            var newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            while (grid.get(newCoords).type != 1 /* EMPTY */) {
                                newCoords = new CartesianCoords(Math.round(Math.random() * grid.gridw), Math.round(Math.random() * grid.gridh));
                            }
                            grid.set(newCoords, t);
                            var canPop = this.floodAcquire(newCoords, t).length == t.value;
                        } while(canPop);
                    }
                }
            }

            // count active cells
            this.gameState.activeCells = 0;
            for (i = 2; i <= gp.maxVal; i++) {
                if (count[i]) {
                    this.gameState.activeCells += count[i];
                    tr.tiles[i] = count[i];
                } else {
                    console.log("bad state:" + i + " is null");
                }
            }
        }
    };

    SquareGame.prototype.justMove = function (tileToMove, direction, game) {
        function displace(set, direction) {
            return set.map(function (cell) {
                return new CartesianCoords(cell.x + direction.x, cell.y + direction.y);
            });
        }

        function checkCollision(newset, oldset) {
            return newset.map(function (cell, i) {
                // if cell is out of bounds, then, collision
                // if cell is not in original set and cell is not -1 then collision
                // if cell is not in original set and cell is -1 then no collision
                // if cell is in original set then no collsion
                var cellIsOutofBounds = game.grid.get(cell).type == 0 /* OUT_OF_BOUNDS */;
                var cellInOldSet = oldset.some(function (c) {
                    return c.x == cell.x && c.y == cell.y;
                });
                var isCollision = cellIsOutofBounds || (!cellInOldSet && game.grid.get(cell).type != 1 /* EMPTY */);
                return isCollision;
            });
        }

        function move(from, to, postCallback) {
            // cache all the from values before clearing them
            var fromVals = from.map(function (cell) {
                return game.grid.get(cell);
            });
            from.forEach(function (cell) {
                game.grid.set(cell, new Tile(1 /* EMPTY */, -1));
            });
            to.forEach(function (cell, i) {
                game.grid.set(cell, new Tile(fromVals[i].type, fromVals[i].value));
            });

            // using a forEach makes sure the i is probably saved in the closure,
            // otherwise, we get multiple calls here to anim.after because the condition for i == 0 is always fulfilled
            from.forEach(function (fromElement, i) {
                var f = game.gridView.getSVGElement(fromElement);
                var t = game.gridView.getSVGElement(to[i]);

                game.gameState.disableMouse = true;

                var anim = f.animate(100, '>', 0).move(t.transform('x'), t.transform('y'));

                if (i == 0) {
                    anim.after(function () {
                        anim.stop();
                        game.gameState.disableMouse = false;
                        game.prune(to[0], function () {
                            game.draw();
                            game.extendUI();
                            game.update();
                            postCallback();
                        });
                    });
                }
            });
        }

        if (game.grid.get(tileToMove).type == 1 /* EMPTY */) {
            return;
        }

        game.prune(tileToMove, function () {
            game.draw();
            game.extendUI();
            game.update();
        });

        if (game.grid.get(tileToMove).type == 1 /* EMPTY */) {
            return;
        }

        var groupToMove = game.floodAcquire(tileToMove, game.grid.get(tileToMove));

        var delta_vec = { x: 0, y: 0 };

        switch (direction) {
            case 0 /* NORTH */:
                delta_vec = { x: 0, y: -1 };
                break;
            case 2 /* SOUTH */:
                delta_vec = { x: 0, y: 1 };
                break;
            case 3 /* WEST */:
                delta_vec = { x: -1, y: 0 };
                break;
            case 1 /* EAST */:
                delta_vec = { x: 1, y: 0 };
                break;
        }

        var oldset = groupToMove;
        var newset = oldset.map(Utils.deepCopy);
        var magnitude = 0;
        while (checkCollision(newset, oldset).every(function (col) {
            return col == false;
        })) {
            oldset = newset.map(Utils.deepCopy); // oldset = newset (deep copy)
            newset = displace(oldset, delta_vec);
            magnitude++;
        }

        var marked = { get: null, set: null };
        marked.get = function (key) {
            return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
        };
        marked.set = function (key) {
            this[JSON.stringify(key)] = true;
        };

        var thresholdValue = game.grid.get(tileToMove).value;
        move(groupToMove, oldset, function () {
            if (magnitude <= 1)
                return;

            // for each check collision between newset and oldset, if true then
            // move that square to the new direction if it's smaller than selected[0]'s number
            checkCollision(newset, oldset).forEach(function (col, i) {
                if (col && game.grid.get(newset[i]).type != 0 /* OUT_OF_BOUNDS */) {
                    if (game.grid.get(newset[i]).value < thresholdValue && !marked.get(newset[i])) {
                        var newnewset = displace(newset, delta_vec);
                        if (game.grid.get(newnewset[i]).type != 0 /* OUT_OF_BOUNDS */ && game.grid.get(newnewset[i]).value > game.grid.get(newset[i]).value) {
                            marked.set(newset[i]);
                            game.crush(newset[i], new TSM.vec2([delta_vec.x, delta_vec.y]), function () {
                                game.justMove(oldset[i], direction, game);
                            });
                        } else {
                            game.floodAcquire(newset[i]).forEach(function (c) {
                                marked.set(c);
                            });
                            game.justMove(newset[i], direction, game);
                        }
                    }
                }
            });
        });
    };

    SquareGame.prototype.crush = function (loc, fromDir, postCallback) {
        var startTile = this.grid.get(loc);
        if (startTile.type == 2 /* REGULAR */) {
            this.grid.set(loc, new Tile(1 /* EMPTY */, -1));
            var c = this.gridView.getSVGElement(loc);
            var game = this;

            // TODO refactor this so game doesn't care about size
            var cellh = this.canvas.width() / Math.sqrt(this.grid.size);
            var horiz = Math.abs(fromDir.x) > 0;
            var vert = !horiz;
            c.animate(100, '<', 0).scale(horiz ? 0 : 1, vert ? 0 : 1).move(horiz ? c.cannonicalTransform.x + fromDir.x * cellh / 2 : c.cannonicalTransform.x, vert ? c.cannonicalTransform.y + fromDir.y * cellh / 2 : c.cannonicalTransform.y).after(function () {
                game.gameState.activeCells--;
                game.tracker.tiles[startTile.value]--;
                console.log("CRUSH:" + game.gameState.activeCells);
                game.gameState.lastCleared = game.grid.get(loc).value;
                postCallback();
            });
        }
    };

    SquareGame.prototype.onDrag = function (e, game) {
        if (game.gameState.selected == null || game.gameState.selected.length == 0) {
            console.log("nothing selected");
            return;
        }

        var moveDirection;

        if (Math.abs(e.gesture.deltaY) > Math.abs(e.gesture.deltaX)) {
            moveDirection = e.gesture.deltaY < 0 ? 0 /* NORTH */ : 2 /* SOUTH */;
        } else {
            moveDirection = e.gesture.deltaX < 0 ? 3 /* WEST */ : 1 /* EAST */;
        }

        function displace(set, direction) {
            return set.map(function (cell) {
                return new CartesianCoords(cell.x + direction.x, cell.y + direction.y);
            });
        }

        function checkCollision(newset, oldset) {
            return newset.map(function (cell, i) {
                // if cell is out of bounds, then, collision
                // if cell is not in original set and cell is not -1 then collision
                // if cell is not in original set and cell is -1 then no collision
                // if cell is in original set then no collsion
                var cellIsOutofBounds = game.grid.get(cell).type == 0 /* OUT_OF_BOUNDS */;
                var cellInOldSet = oldset.some(function (c) {
                    return c.x == cell.x && c.y == cell.y;
                });
                var isCollision = cellIsOutofBounds || (!cellInOldSet && game.grid.get(cell).type != 1 /* EMPTY */);
                return isCollision;
            });
        }

        function move(from, to, postCallback) {
            // cache all the from values before clearing them
            var fromVals = from.map(function (cell) {
                return game.grid.get(cell);
            });
            from.forEach(function (cell) {
                game.grid.set(cell, new Tile(1 /* EMPTY */, -1));
            });
            to.forEach(function (cell, i) {
                game.grid.set(cell, new Tile(fromVals[i].type, fromVals[i].value));
            });

            game.gameState.disableMouse = true;

            // using a forEach makes sure the i is probably saved in the closure,
            // otherwise, we get multiple calls here to anim.after because the condition for i == 0 is always fulfilled
            from.forEach(function (fromElement, i) {
                var f = game.gridView.getSVGElement(fromElement);
                var t = game.gridView.getSVGElement(to[i]);

                var anim = f.animate(100, '>', 0).move(t.transform('x'), t.transform('y'));

                if (i == 0) {
                    anim.after(function () {
                        anim.stop();
                        game.gameState.disableMouse = false;
                        game.prune(to[0], function () {
                            game.draw();
                            game.extendUI();
                            game.update();
                            postCallback();
                        });
                    });
                }
            });
        }

        if (game.grid.get(game.gameState.selected[0]).type == 1 /* EMPTY */) {
            return;
        }

        game.prune(game.gameState.selected[0], function () {
            game.draw();
            game.extendUI();
            game.update();
        });

        if (game.grid.get(game.gameState.selected[0]).type == 1 /* EMPTY */) {
            return;
        }

        var delta_vec = { x: 0, y: -1 };
        switch (moveDirection) {
            case 0 /* NORTH */:
                delta_vec = { x: 0, y: -1 };
                break;
            case 1 /* EAST */:
                delta_vec = { x: 1, y: 0 };
                break;
            case 2 /* SOUTH */:
                delta_vec = { x: 0, y: 1 };
                break;
            case 3 /* WEST */:
                delta_vec = { x: -1, y: 0 };
                break;
        }

        var marked = { get: null, set: null };
        marked.get = function (key) {
            return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
        };
        marked.set = function (key) {
            this[JSON.stringify(key)] = true;
        };

        var oldset = game.gameState.selected.map(Utils.deepCopy);
        var newset = displace(oldset, delta_vec);
        var magnitude = 0;
        while (checkCollision(newset, oldset).every(function (col) {
            return col == false;
        })) {
            oldset = newset.map(Utils.deepCopy); // oldset = newset (deep copy)
            newset = displace(oldset, delta_vec);
            magnitude++;
        }

        var thresholdValue = game.grid.get(game.gameState.selected[0]).value;

        move(game.gameState.selected, oldset, function () {
            if (magnitude < 1)
                return;
            game.gameState.numMoves++; // hidden property; hard to find, understand, and debug

            // for each check collision between newset and oldset, if true then
            // move that square to the new direction if it's smaller than selected[0]'s number
            checkCollision(newset, oldset).forEach(function (col, i) {
                if (col && game.grid.get(newset[i]).type != 0 /* OUT_OF_BOUNDS */) {
                    if (game.grid.get(newset[i]).value < thresholdValue && !marked.get(newset[i])) {
                        var newnewset = displace(newset, delta_vec);
                        if (game.grid.get(newnewset[i]).type != 0 /* OUT_OF_BOUNDS */ && game.grid.get(newnewset[i]).value > game.grid.get(newset[i]).value) {
                            marked.set(newset[i]);
                            game.crush(newset[i], new TSM.vec2([delta_vec.x, delta_vec.y]), function () {
                                game.justMove(oldset[i], moveDirection, game);
                            });
                        } else {
                            game.floodAcquire(newset[i]).forEach(function (c) {
                                marked.set(c);
                            });
                            game.justMove(newset[i], moveDirection, game);
                        }
                    }
                }
            });
        });

        game.gameState.selected = oldset; // shallow copy is fine
    };

    SquareGame.prototype.over = function () {
        return this.gameState.activeCells >= this.grid.size - 1;
    };

    SquareGame.prototype.clearedStage = function () {
        return this.gameState.activeCells == 0;
    };

    SquareGame.prototype.advance = function () {
        var gp = Utils.deepCopy(this.gameParams);
        gp.level++;

        gp.gridw = [5, 8, 10, 16, 20].reduce(function (prev, cur, _, __) {
            if (prev > gp.gridw)
                return prev;
            else if (cur > gp.gridw)
                return cur;
        });
        gp.gridh = gp.gridw;

        gp.maxVal = Math.floor(Math.sqrt(gp.gridw * gp.gridh));
        if (gp.maxVal > 9)
            gp.maxVal = 9;

        this.gridView.updateTimerBar(null, null, null);
        this.init(gp);
    };

    SquareGame.prototype.floodAcquire = function (start, tile) {
        var cluster = [];
        var marked = { get: null, set: null };
        marked.get = function (key) {
            return this[JSON.stringify(key)] === undefined ? false : this[JSON.stringify(key)];
        };
        marked.set = function (key) {
            this[JSON.stringify(key)] = true;
        };
        var Q = [];
        if (this.grid.get(new CartesianCoords(start.x, start.y)) != tile)
            return [];
        Q.push(start);
        while (Q.length > 0) {
            var n = Q.shift();
            if (this.grid.get(n).value == tile.value && this.grid.get(n).type == tile.type && !marked.get(n)) {
                var w = new CartesianCoords(n.x, n.y);
                var e = new CartesianCoords(n.x, n.y);

                while (this.grid.get(new CartesianCoords(w.x - 1, w.y)).value == tile.value && this.grid.get(new CartesianCoords(w.x - 1, w.y)).type == tile.type) {
                    w.x--;
                }

                while (this.grid.get(new CartesianCoords(e.x + 1, e.y)).value == tile.value && this.grid.get(new CartesianCoords(e.x + 1, e.y)).type == tile.type) {
                    e.x++;
                }

                for (var x = w.x; x < e.x + 1; x++) {
                    var nn = new CartesianCoords(x, n.y);
                    marked.set(nn);
                    cluster.push(nn);
                    var north = new CartesianCoords(nn.x, nn.y - 1);
                    var south = new CartesianCoords(nn.x, nn.y + 1);
                    if (this.grid.get(north).value == tile.value && this.grid.get(north).type == tile.type)
                        Q.push(north);
                    if (this.grid.get(south).value == tile.value && this.grid.get(south).type == tile.type)
                        Q.push(south);
                }
            }
        }
        return cluster;
    };

    SquareGame.prototype.prune = function (start, postCallback) {
        this.draw(); // sync model and view/DOM elements -- this is important, otherwise the view will be outdated and animations won't play right!!!

        // see if we should delete this cell and surrounding cells
        var startTile = this.grid.get(start);
        var targets = this.floodAcquire(start, startTile);
        if (targets.length >= startTile.value) {
            console.log("pruning " + targets.length + " tiles");
            if (startTile.type == 4 /* DEACTIVATED */)
                return;
            var game = this;
            targets.forEach(function (cell, i) {
                if (i >= startTile.value)
                    return;
                if (game.grid.get(cell).type == 2 /* REGULAR */) {
                    game.grid.set(cell, new Tile(1 /* EMPTY */, -1));
                    var c = game.gridView.getSVGElement(cell);
                    c.animate(200, '>', 0).scale(0, 0).after(function () {
                        c.stop();
                        game.gameState.activeCells--;
                        game.tracker.tiles[startTile.value]--;
                        console.log("PRUNE:" + game.gameState.activeCells);
                        if (i == startTile.value - 1) {
                            postCallback();
                            game.gameState.lastCleared = startTile.value;
                        }
                    });
                } else if (game.grid.get(cell).type == 5 /* LAVA */) {
                    game.grid.set(cell, new Tile(4 /* DEACTIVATED */, game.grid.get(cell).value));
                }
            });
        } else {
            postCallback();
        }
    };
    return SquareGame;
})();
/// <reference path="./hexgame.ts" />
/// <reference path="./squaregame.ts" />
/// <reference path="lib/utils.ts" />
var game;

var init = function () {
    var doSurvival = Utils.getURLParameter("survival") == "true";
    var useHexGrid = Utils.getURLParameter("hex") == "true";

    var canvas = SVG('screen').size(640, 650);
    var gameMode = doSurvival ? 0 /* SURVIVAL */ : 1 /* PUZZLE */;

    game = useHexGrid ? new HexGame(gameMode, canvas) : new SquareGame(gameMode, canvas);
};
